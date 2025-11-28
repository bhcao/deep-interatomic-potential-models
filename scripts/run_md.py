# Copyright 2025 Cao Bohan
#
# DIPM is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DIPM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
from pathlib import Path
import logging
from typing import Literal

from ase import io, Atoms
from dipm.utils.model_io import load_model
from dipm.simulation import JaxMDSimulationEngine, SimulationState

logger = logging.getLogger("dipm")


class MDLogger:
    def __init__(
        self,
        names: list[str],
        structures: list[Atoms],
        root: Path = Path("results"),
        ext: Literal["pdb", "extxyz"] = "pdb",
    ):
        self.structures = structures
        self.logfiles = []
        self.pdbfiles = []
        self.format = {"pdb": "proteindatabank", "extxyz": "extxyz"}[ext]
        for name, atoms in zip(names, structures):
            logfile = open(root / f"log_{name}.txt", "w", encoding="utf-8")
            logfile.write(f"{'Step':>10} {'Ekin[eV]':>12} {'T[K]':>6}\n")
            self.logfiles.append(logfile)
            pdbfile = open(root / f"trajectory_{name}.{ext}", "w", encoding="utf-8")
            io.write(pdbfile, atoms, format=self.format)
            self.pdbfiles.append(pdbfile)
        self.fmt = "%10d %12.4f %6.1f\n"

    def __del__(self):
        for logfile in self.logfiles:
            logfile.close()
        for pdbfile in self.pdbfiles:
            pdbfile.close()

    def __call__(self, state: SimulationState):
        for logfile, pdbfile, position, e_kin, temp, structure in zip(
            self.logfiles, self.pdbfiles, state.positions, state.kinetic_energy,
            state.temperature, self.structures
        ):
            frame = structure.copy()
            frame.set_positions(position[-1])
            io.write(pdbfile, frame, format=self.format)
            pdbfile.flush()
            logfile.write(self.fmt % (state.step, e_kin[-1], temp[-1]))
            logfile.flush()

# pylint: disable=redefined-outer-name
def main(args):
    logger.info("Loading model...")

    force_field = load_model(args.model)

    logger.info("Loading structures from files...")

    structures = []
    structures_names = []

    path = Path(args.path)
    if path.is_file():
        structures.append(io.read(path))
        structures_names.append(path.name)
    else:
        for name in os.listdir(path):
            if (path / name).suffix in ['.pdb', '.xyz', '.extxyz']:
                structures.append(io.read(path / name))
                structures_names.append((path / name).stem)

    num_steps = int(round(args.run_ns * 1e6 / args.timestep_fs))
    log_interval = int(round(args.log_interval_fs / args.timestep_fs))
    num_episodes = num_steps // log_interval
    config = JaxMDSimulationEngine.Config(
        timestep_fs=args.timestep_fs,
        num_steps=num_steps,
        num_episodes=num_episodes,
        snapshot_interval=log_interval,
        temperature_kelvin=args.temperature,
    )

    batch_size = args.batch_size or len(structures)
    len_batch = len(structures) // batch_size
    if len(structures) % batch_size != 0:
        len_batch += 1

    logger.info(
        "Running simulation for %s structures spilt into %s batches...", len(structures), len_batch
    )
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    for i, i_batch in enumerate(range(0, len(structures), batch_size)):

        logger.info("Initializing simulation engine for batch %s...", i+1)
        batch_structures = structures[i_batch:min(i_batch+batch_size, len(structures))]
        batch_structures_names = structures_names[i_batch:min(i_batch+batch_size, len(structures))]
        md_engine = JaxMDSimulationEngine(batch_structures, force_field, config)
        md_logger = MDLogger(
            batch_structures_names, batch_structures, log_dir, args.extention
        )
        md_engine.attach_logger(md_logger)

        logger.info("Running simulation for batch %s...", i+1)
        md_engine.run()

        logger.info("Writing final structures for batch %s...", i+1)
        for name, positions, structure in zip(
            batch_structures_names, md_engine.state.positions, batch_structures
        ):
            final_atoms = structure.copy()
            final_atoms.set_positions(positions[-1])
            io.write(
                log_dir / f"final_structure_{name}.{args.extention}",
                final_atoms,
                format=md_logger.format,
            )

        logger.info("Batch %s completed.", i+1)
    logger.info("Simulation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model to use for the simulation.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the file/folder containing initial structures.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of structures to run in parallel, default is all.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results",
        help="Directory to store logs and results.",
    )
    parser.add_argument(
        "--extention",
        type=str,
        default="pdb",
        choices=["pdb", "extxyz"],
        help="Extension of the output files, default is pdb.",
    )
    parser.add_argument(
        "--run_ns",
        type=float,
        default=1.0,
        help="Total time of the simulation in ns, default is 1ns.",
    )
    parser.add_argument(
        "--timestep_fs",
        type=float,
        default=1.0,
        help="Timestep of the simulation in fs, default is 1fs.",
    )
    parser.add_argument(
        "--log_interval_fs",
        type=float,
        default=100.0,
        help="Interval of logging the simulation in fs, default is 100fs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature of the simulation in Kelvin, default is 300K.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, force=True, format='%(levelname)s (%(name)s): %(message)s'
    )
    logger.setLevel(logging.INFO)

    main(args)
