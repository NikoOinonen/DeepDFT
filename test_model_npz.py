import logging
import os
import json
import argparse
import math
import contextlib
import pickle
import timeit

import ase
import ase.io
import torch
import numpy as np

import dataset
import densitymodel

from pathlib import Path
from torch.utils.data import Dataset
from ase.io.xsf import write_xsf


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Predict with pretrained model", fromfile_prefix_chars="@")
    parser.add_argument("model_dir", type=str, help="Directory of pretrained model")
    parser.add_argument("--probe_count", type=int, default=5000, help="How many probe points to compute per iteration")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for inference e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, disable periodic boundary conditions (force to False) in atoms data",
    )
    parser.add_argument(
        "--force_pbc",
        action="store_true",
        help="If flag is given, force periodic boundary conditions to True in atoms data",
    )

    return parser.parse_args(arg_list)


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "arguments.json"), "r") as f:
        runner_args = argparse.Namespace(**json.load(f))
    if runner_args.use_painn_model:
        model = densitymodel.PainnDensityModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff)
    else:
        model = densitymodel.DensityModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff)
    device = torch.device(device)
    model.to(device)
    state_dict = torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state_dict["model"])
    return model, runner_args.cutoff


class LazyMeshGrid:
    def __init__(self, cell, grid_step, origin=None, adjust_grid_step=False):
        self.cell = cell
        if adjust_grid_step:
            n_steps = np.round(self.cell.lengths() / grid_step)
            self.scaled_grid_vectors = [np.arange(n) / n for n in n_steps]
            self.adjusted_grid_step = self.cell.lengths() / n_steps
        else:
            self.scaled_grid_vectors = [np.arange(0, l, grid_step) / l for l in self.cell.lengths()]
        self.shape = np.array([len(g) for g in self.scaled_grid_vectors] + [3])
        if origin is None:
            self.origin = np.zeros(3)
        else:
            self.origin = origin

        self.origin = np.expand_dims(self.origin, 0)

    def __getitem__(self, indices):
        indices = np.array(indices)
        indices_shape = indices.shape
        if not (len(indices_shape) == 2 and indices_shape[0] == 3):
            raise NotImplementedError("Indexing must be a 3xN array-like object")
        gridA = self.scaled_grid_vectors[0][indices[0]]
        gridB = self.scaled_grid_vectors[1][indices[1]]
        gridC = self.scaled_grid_vectors[2][indices[2]]

        grid_pos = np.stack([gridA, gridB, gridC], 1)
        grid_pos = np.dot(grid_pos, self.cell)
        grid_pos += self.origin

        return grid_pos


class DensityDataset(Dataset):

    def __init__(self, data_dir: Path, sample_paths: list[Path]):
        self.sample_paths = [data_dir / p for p in sample_paths]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:

        sample_path = self.sample_paths[index]
        sample_npz = np.load(sample_path)
        ref_density = sample_npz["data"]
        xyzs = sample_npz["xyz"]
        Zs = sample_npz["Z"]
        lattice = sample_npz["lattice"]
        origin = sample_npz["origin"]
        sample_npz.close()

        xyzs -= origin
        grid_step = lattice[0, 0] / ref_density.shape[0]

        atoms = ase.Atoms(numbers=Zs, positions=xyzs, cell=lattice)
        grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step)
        origin = np.zeros(3)

        sample = {
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "ref_density": ref_density,
            "metadat": {"filename": sample_path},
        }

        return sample


def main():
    args = get_arguments()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("printlog.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )

    model, cutoff = load_model(args.model_dir, args.device)

    sample_dict_path = Path("./sample_dict.pickle")
    data_dir = Path("/mnt/triton/density_db")
    with open(sample_dict_path, "rb") as f:
        sample_dict = pickle.load(f)
        density_dataset = DensityDataset(data_dir, sample_dict["test"])

    for density_dict in density_dataset:

        device = torch.device(args.device)

        if args.ignore_pbc and args.force_pbc:
            raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
        elif args.ignore_pbc:
            set_pbc = False
        elif args.force_pbc:
            set_pbc = True
        else:
            set_pbc = None

        start_time = timeit.default_timer()

        with torch.no_grad():

            # Make graph with no probes
            logging.debug("Computing atom-to-atom graph")
            collate_fn = dataset.CollateFuncAtoms(
                cutoff=cutoff,
                pin_memory=device.type == "cuda",
                set_pbc_to=set_pbc,
            )
            graph_dict = collate_fn([density_dict])
            logging.debug("Computing atom representation")
            device_batch = {k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()}
            if isinstance(model, densitymodel.PainnDensityModel):
                atom_representation_scalar, atom_representation_vector = model.atom_model(device_batch)
            else:
                atom_representation = model.atom_model(device_batch)
            logging.debug("Atom representation done")

            # Loop over all slices
            pred_density = []
            for probe_graph_dict in dataset.DensityGridIterator(density_dict, args.probe_count, cutoff, set_pbc_to=set_pbc):

                probe_dict = dataset.collate_list_of_dicts([probe_graph_dict])
                probe_dict = {k: v.to(device=device, non_blocking=True) for k, v in probe_dict.items()}
                device_batch["probe_edges"] = probe_dict["probe_edges"]
                device_batch["probe_edges_displacement"] = probe_dict["probe_edges_displacement"]
                device_batch["probe_xyz"] = probe_dict["probe_xyz"]
                device_batch["num_probe_edges"] = probe_dict["num_probe_edges"]
                device_batch["num_probes"] = probe_dict["num_probes"]

                if isinstance(model, densitymodel.PainnDensityModel):
                    density = model.probe_model(
                        device_batch,
                        atom_representation_scalar,
                        atom_representation_vector,
                        compute_iri=False,
                        compute_dori=False,
                        compute_hessian=False,
                    )
                else:
                    density = model.probe_model(
                        device_batch,
                        atom_representation,
                        compute_iri=False,
                        compute_dori=False,
                        compute_hessian=False,
                    )

                pred_density.append(density)

            ref_density = density_dict["ref_density"]
            ref_density = torch.from_numpy(ref_density).to(device)

            pred_density = torch.cat(pred_density, dim=1)
            pred_density = pred_density.reshape(ref_density.shape)

            diff = ref_density - pred_density
            mse = (diff**2).mean()
            print(mse)

            atoms = density_dict["atoms"]
            atoms.set_pbc(True)
            with open("test_pred.xsf", "w") as f:
                write_xsf(f, [atoms], data=pred_density.cpu().numpy())
            with open("test_ref.xsf", "w") as f:
                write_xsf(f, [atoms], data=ref_density.cpu().numpy())
            with open("test_diff.xsf", "w") as f:
                write_xsf(f, [atoms], data=diff.cpu().numpy())

            exit()

        end_time = timeit.default_timer()

        logging.info("done time_elapsed=%f", end_time - start_time)


if __name__ == "__main__":
    main()
