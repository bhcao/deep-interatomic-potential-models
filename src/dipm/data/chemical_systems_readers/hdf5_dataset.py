# Copyright 2025 Cao Bohan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
import os

import h5py
import numpy as np

from dipm.data.chemical_system import ChemicalSystem
from dipm.data.chemical_systems_readers.dataset import Dataset, ConcatDataset, Subset
from dipm.data.configs import ChemicalSystemsReaderConfig

DEFAULT_WEIGHT = 1.0
DEFAULT_PBC = np.zeros(3, bool)
DEFAULT_CELL = np.zeros((3, 3))


# Fuck MACE! Why not represent None by not providing a key?
def _unpack_value(value):
    '''In MACE h5 dataset, None is transformed to str.'''
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


class Hdf5Dataset(Dataset):
    """Loads data from a single hdf5 file."""

    def __init__(self, file_path: os.PathLike, shuffle: bool = False):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            batch_key = list(f.keys())[0]
            self.batch_size = len(f[batch_key].keys())
            self.length = len(f.keys()) * self.batch_size
        self._file = None # lazy load for multiprocessing safety
        self.shuffle = shuffle
        if self.shuffle:
            self.indices = np.random.permutation(self.length)

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")

        if self.shuffle:
            index = self.indices[index]

        if isinstance(index, slice):
            return [self._get_item(i) for i in range(
                index.start or 0, index.stop or len(self), index.step or 1
            )]
        if isinstance(index, (list, np.ndarray)):
            return [self._get_item(i) for i in index]

        return self._get_item(index)

    def _get_item(self, index: int) -> ChemicalSystem:
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

        pbc = _unpack_value(subgrp["pbc"][()])
        cell = _unpack_value(subgrp["cell"][()])
        if cell is not None:
            assert np.linalg.det(cell) >= 0.0

        return ChemicalSystem(
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
        )

    def __len__(self) -> int:
        return self.length

    def release(self):
        '''Release dataset file handles.'''
        if self._file is not None:
            self._file.close()
            self._file = None


def _get_num_to_load(num_to_load, dataset_size):
    '''Returns the number of data points to load from a dataset.'''
    if num_to_load is None:
        return dataset_size
    if isinstance(num_to_load, float):
        return int(dataset_size * num_to_load)
    return num_to_load


def create_datasets(
    config: ChemicalSystemsReaderConfig, post_process_fn: Callable | None = None
) -> tuple[ConcatDataset, ConcatDataset | None, ConcatDataset | None]:
    '''It's recommended to call release before switching dataset.'''

    train_datasets = ConcatDataset([
        Hdf5Dataset(ds_path) for ds_path in config.train_dataset_paths
    ], shuffle=config.shuffle, parallel=config.parallel, post_process_fn=post_process_fn)

    if config.dataset_splits is not None:
        # Handle dataset splits.
        splits = config.dataset_splits
        if isinstance(splits[0], float):
            dataset_size = len(train_datasets)
            val_size = int(dataset_size * splits[1])
            test_size = int(dataset_size * splits[2])
            # Ensure all data is used when splits are added up to 1.0.
            if splits[0] + splits[1] + splits[2] == 1.0:
                train_size = dataset_size - val_size - test_size
            else:
                train_size = int(dataset_size * splits[0])
            splits = (train_size, val_size, test_size)

        # Handle num_to_load.
        splits_lens = (
            _get_num_to_load(config.train_num_to_load, splits[0]),
            _get_num_to_load(config.valid_num_to_load, splits[1]),
            _get_num_to_load(config.test_num_to_load, splits[2]),
        )

        return (
            Subset(train_datasets, 0, splits_lens[0]),
            Subset(train_datasets, splits[0], splits_lens[1]),
            Subset(train_datasets, splits[0] + splits[1], splits_lens[2])
        )

    if config.train_num_to_load is not None:
        num_to_load = _get_num_to_load(config.train_num_to_load, len(train_datasets))
        train_datasets = Subset(train_datasets, 0, num_to_load)

    valid_datasets = None
    if config.valid_dataset_paths is not None:
        valid_datasets = ConcatDataset([
            Hdf5Dataset(ds_path) for ds_path in config.valid_dataset_paths
        ], shuffle=config.shuffle, parallel=config.parallel, post_process_fn=post_process_fn)
        if config.valid_num_to_load is not None:
            num_to_load = _get_num_to_load(
                config.valid_num_to_load, len(valid_datasets)
            )
            valid_datasets = Subset(valid_datasets, 0, num_to_load)

    test_datasets = None
    if config.test_dataset_paths is not None:
        test_datasets = ConcatDataset([
            Hdf5Dataset(ds_path) for ds_path in config.test_dataset_paths
        ], shuffle=config.shuffle, parallel=config.parallel, post_process_fn=post_process_fn)
        if config.test_num_to_load is not None:
            num_to_load = _get_num_to_load(
                config.test_num_to_load, len(test_datasets)
            )
            test_datasets = Subset(test_datasets, 0, num_to_load)

    return train_datasets, valid_datasets, test_datasets
