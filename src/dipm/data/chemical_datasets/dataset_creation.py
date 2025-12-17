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
from pathlib import Path
from typing import Any

import numpy as np

from dipm.data.chemical_datasets.dataset import ConcatDataset, Subset
from dipm.data.chemical_system import ChemicalSystem
from dipm.data.chemical_datasets.hdf5_dataset import Hdf5Dataset
from dipm.data.configs import DatasetCreationConfig

_PostProcessFn = Callable[[ChemicalSystem], Any] | None

def _get_num_to_load(num_to_load, dataset_size):
    """Returns the number of data points to load from a dataset."""
    if num_to_load is None:
        return dataset_size
    if isinstance(num_to_load, float):
        return int(dataset_size * num_to_load)
    return num_to_load


class DatasetCreation:
    """Class for creating datasets.
    
    Attributes:
        post_process_fn (Callable[[ChemicalSystem], Any] | None): Function to apply to each
            system after loading.
    """

    Config = DatasetCreationConfig

    def __init__(self, config: DatasetCreationConfig):
        self.config = config
        self.task_list = (
            list(config.train_dataset_paths.keys())
            if isinstance(config.train_dataset_paths, dict) else None
        )

    def _get_datasets(
        self,
        paths: list[Path] | dict[str, list[Path]],
        post_process_fn: _PostProcessFn = None,
        load_exclusions: bool = True,
    ) -> list[Hdf5Dataset]:
        """Create list of dataset classes from paths."""

        datasets = []
        if isinstance(paths, list):
            for p in paths:
                exclude_path = p.with_name(p.stem + '_exclude.npy')
                exclude_ids = None
                if load_exclusions and exclude_path.exists():
                    exclude_ids = np.load(exclude_path)
                datasets.append(Hdf5Dataset(p, exclude_ids, None, post_process_fn))
            return datasets

        if self.task_list is None:
            raise ValueError("Task list must be provided for multi-task datasets.")

        for task, path in paths.items():
            for p in path:
                exclude_path = p.with_name(p.stem + '_exclude.npy')
                exclude_ids = None
                if load_exclusions and exclude_path.exists():
                    exclude_ids = np.load(exclude_path)
                datasets.append(
                    Hdf5Dataset(p, exclude_ids, self.task_list.index(task), post_process_fn)
                )

        return datasets

    def get_train_datasets(
        self, post_process_fn: _PostProcessFn = None, load_exclusions: bool = True
    ) -> list[Hdf5Dataset]:
        """List of training datasets."""
        return self._get_datasets(
            self.config.train_dataset_paths, post_process_fn, load_exclusions
        )

    def get_valid_datasets(
        self, post_process_fn: _PostProcessFn = None, load_exclusions: bool = True
    ) -> list[Hdf5Dataset]:
        """List of validation datasets."""
        if self.config.valid_dataset_paths is None:
            return []
        return self._get_datasets(
            self.config.valid_dataset_paths, post_process_fn, load_exclusions
        )

    def get_test_datasets(
        self, post_process_fn: _PostProcessFn = None, load_exclusions: bool = True
    ) -> list[Hdf5Dataset]:
        """List of test datasets."""
        if self.config.test_dataset_paths is None:
            return []
        return self._get_datasets(self.config.test_dataset_paths, post_process_fn, load_exclusions)

    def _split_dataset(self, dataset):
        """Split a dataset into training, validation, and test sets."""

        splits = self.config.dataset_splits
        if isinstance(splits[0], float):
            dataset_size = len(dataset)
            val_size = int(dataset_size * splits[1])
            test_size = int(dataset_size * splits[2])
            # Ensure all data is used when splits are added up to 1.0.
            if splits[0] + splits[1] + splits[2] == 1.0:
                train_size = dataset_size - val_size - test_size
            else:
                train_size = int(dataset_size * splits[0])
            splits = (train_size, val_size, test_size)

        return (
            self._get_subset(dataset, self.config.train_num_to_load, 0, splits[0]),
            self._get_subset(dataset, self.config.valid_num_to_load, splits[0], splits[1]),
            self._get_subset(
                dataset, self.config.test_num_to_load, splits[0] + splits[1], splits[2]
            ),
        )

    def create_datasets(
        self, post_process_fn: _PostProcessFn = None
    ) -> tuple[ConcatDataset, ConcatDataset | None, ConcatDataset | None]:
        """Create dataset from config."""

        train_dataset = ConcatDataset(self.get_train_datasets(post_process_fn))

        if self.config.dataset_splits is not None:
            return self._split_dataset(train_dataset)

        train_dataset = self._get_subset(train_dataset, self.config.train_num_to_load)

        valid_dataset = None
        if self.config.valid_dataset_paths is not None:
            valid_dataset = ConcatDataset(self.get_valid_datasets(post_process_fn))
            valid_dataset = self._get_subset(valid_dataset, self.config.valid_num_to_load)

        test_dataset = None
        if self.config.test_dataset_paths is not None:
            test_dataset = ConcatDataset(self.get_test_datasets(post_process_fn))
            test_dataset = self._get_subset(test_dataset, self.config.test_num_to_load)

        return train_dataset, valid_dataset, test_dataset

    def _get_subset(self, dataset, num_to_load, start=0, length=None):
        """Get a subset from a slice of a dataset. If ``random_subset`` is ``False``,
        load the first ``num_to_load`` data points. Otherwise, randomly select
        ``num_to_load`` data points from the dataset slice.

        ``subset = sample(dataset[start:start+length], num_to_load)``

        Args:
            dataset (ConcatDataset): Dataset to get subset from.
            num_to_load (int | float | None): Number of data points to load.
            start (int, optional): Start index of slice. Defaults to 0.
            length (int | None, optional): Length of slice. Defaults to len(dataset).
        """

        if num_to_load is None:
            return dataset

        if length is None:
            length = len(dataset)

        num_to_load = _get_num_to_load(num_to_load, length)

        if self.config.random_subset:
            choices = np.sort(np.random.choice(length, num_to_load, replace=False))
            return Subset(dataset, start + choices)

        return Subset(dataset, start, num_to_load)

    @staticmethod
    def filter_duplicates(datasets: list[Hdf5Dataset], datasets_to_exclude: list[Hdf5Dataset]):
        """Filter out duplicate datasets."""
        paths_to_exclude = set(dataset.path for dataset in datasets_to_exclude)
        return [
            dataset for dataset in datasets
            if dataset.path not in paths_to_exclude
        ]
