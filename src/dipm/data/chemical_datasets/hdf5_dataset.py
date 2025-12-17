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

from collections.abc import Callable
import os
from typing import TypeVar

import h5py
import numpy as np

from dipm.data.chemical_system import ChemicalSystem
from dipm.data.chemical_datasets.dataset import Dataset

DEFAULT_WEIGHT = 1.0
DEFAULT_PBC = np.zeros(3, bool)
DEFAULT_CELL = np.zeros((3, 3))

_T_co = TypeVar("_T_co", covariant=True)


# Fuck MACE! Why not represent None by not providing a key?
def _unpack_value(value):
    '''In MACE h5 dataset, None is transformed to str.'''
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


class Hdf5Dataset(Dataset[_T_co]):
    """Loads data from a single hdf5 file."""

    def __init__(
        self,
        path: os.PathLike,
        exclude_ids: np.ndarray | None = None,
        task: int | None = None,
        post_process_fn: Callable[[ChemicalSystem], _T_co] | None = None,
    ):
        self.path = path
        with h5py.File(self.path, "r") as f:
            batch_key = list(f.keys())[0]
            self.batch_size = len(f[batch_key].keys())
            self.length = len(f.keys()) * self.batch_size
        self._file = None # lazy load for multiprocessing safety
        self.post_process_fn = post_process_fn
        self.task = task
        if exclude_ids is not None:
            mask = np.ones(self.length, bool)
            mask[exclude_ids] = False
            self.length -= len(exclude_ids)
            self.remap_ids = np.where(mask)[0]
        else:
            self.remap_ids = None

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.path, "r")

        if self.remap_ids is not None:
            index = self.remap_ids[index]

        if isinstance(index, slice):
            return [self._get_item(i) for i in range(
                index.start or 0, index.stop or len(self), index.step or 1
            )]
        if isinstance(index, (list, np.ndarray)):
            return [self._get_item(i) for i in index]

        return self._get_item(index)

    def _get_item(self, index: int) -> _T_co:
        # compute the index of the batch
        batch_index = index // self.batch_size
        config_index = index % self.batch_size
        grp = self._file["config_batch_" + str(batch_index)]
        subgrp = grp["config_" + str(config_index)]

        # extract the data from the hdf5 file
        positions = subgrp["positions"][()]
        atomic_numbers = subgrp["atomic_numbers"][()]

        forces = (
            _unpack_value(subgrp["properties"]["forces"][()])
            if "forces" in subgrp["properties"] else None
        )
        energy = (
            _unpack_value(subgrp["properties"]["energy"][()])
            if "energy" in subgrp["properties"] else None
        )
        stress = (
            _unpack_value(subgrp["properties"]["stress"][()])
            if "stress" in subgrp["properties"] else None
        )
        charges = (
            _unpack_value(subgrp["properties"]["charges"][()])
            if "charges" in subgrp["properties"] else None
        )
        total_charge = (
            _unpack_value(subgrp["properties"]["total_charge"][()])
            if "total_charge" in subgrp["properties"] else None
        )
        total_spin = (
            _unpack_value(subgrp["properties"]["total_spin"][()])
            if "total_spin" in subgrp["properties"] else None
        )
        dipole = (
            _unpack_value(subgrp["properties"]["dipole"][()])
            if "dipole" in subgrp["properties"] else None
        )

        pbc = _unpack_value(subgrp["pbc"][()])
        cell = _unpack_value(subgrp["cell"][()])
        if cell is not None:
            assert np.linalg.det(cell) >= 0.0

        system = ChemicalSystem(
            atomic_numbers=atomic_numbers,
            # will be populated later
            atomic_species=np.empty(atomic_numbers.shape[0]),
            positions=positions,
            energy=energy,
            forces=forces,
            stress=stress,
            cell=cell if cell is not None else DEFAULT_CELL,
            pbc=pbc if pbc is not None else DEFAULT_PBC,
            weight=DEFAULT_WEIGHT,
            atomic_charges=charges,
            charge=total_charge,
            spin=total_spin,
            dipole=dipole,
            task=self.task,
        )
        if self.post_process_fn is not None:
            system = self.post_process_fn(system)
        return system

    def __len__(self) -> int:
        return self.length

    def release(self):
        '''Release dataset file handles.'''
        if self._file is not None:
            self._file.close()
            self._file = None
