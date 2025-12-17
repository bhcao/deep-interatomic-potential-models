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

from collections.abc import Sequence
import logging
from typing import overload, TypeVar, Generic

import numpy as np

logger = logging.getLogger('dipm')

_T_co = TypeVar("_T_co", covariant=True)


class Dataset(Generic[_T_co]):
    """Pytorch dataset like base objects."""

    @overload
    def __getitem__(self, index: int) -> _T_co:...
    @overload
    def __getitem__(self, index: list | np.ndarray | slice) -> list[_T_co]:...
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")

    def release(self):
        """Release resources."""

    def __del__(self):
        self.release()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets. Pytorch-like ConcatDataset."""

    datasets: list[Dataset[_T_co]]

    def __init__(self, datasets: Sequence[Dataset]):
        """
        Create a concatenated dataset.

        Args:
            datasets (sequence): List of datasets to concatenate.
        """
        self.datasets = datasets
        self.lengths = np.array([len(d) for d in self.datasets])
        self.cum_lengths = np.cumsum(self.lengths)
        self.length = int(self.cum_lengths[-1])

        # parallel loader, will be set by DataLoader
        self._loader = None

    def _load(self, indices, ds_indices):
        """Load data from the datasets in parallel or sequentially."""
        if self._loader is not None:
            return self._loader(zip(ds_indices, indices))

        # sequential
        results = []
        for idx, ds_idx in zip(indices, ds_indices):
            data = self.datasets[ds_idx][idx]
            results.append(data)
        return results

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._get_items_by_slice(
                index.start or 0, index.stop or len(self), index.step or 1
            )

        if isinstance(index, (list, np.ndarray)):
            indices = np.array(index, dtype=int)
            return self._get_items_by_indices(indices)

        return self._get_item(index)

    def _get_item(self, index: int) -> _T_co:
        '''Get one item by index.'''
        dataset_idx = np.searchsorted(self.cum_lengths, index, side="right")
        if dataset_idx == 0:
            local_idx = index
        else:
            local_idx = index - self.cum_lengths[dataset_idx - 1]
        return self._load([local_idx], [dataset_idx])[0]

    def _get_items_by_indices(self, indices: np.ndarray) -> list[_T_co]:
        """Batch version: load items dataset by dataset (fewer file seeks)."""
        dataset_indices = np.searchsorted(self.cum_lengths, indices, side="right")

        ds_indices = []
        local_indices = []
        positions = []
        for ds_idx in range(len(self.datasets)):
            mask = dataset_indices == ds_idx
            if not np.any(mask):
                continue
            local_idx = indices[mask]
            if ds_idx > 0:
                local_idx = local_idx - self.cum_lengths[ds_idx - 1]
            ds_indices.append(ds_idx)
            local_indices.append(local_idx)
            positions.append(np.where(mask)[0])

        data = self._load(local_indices, ds_indices)

        result = [None] * len(indices)
        # fill in the result array in the correct order
        for data_pds, positions_pds in zip(data, positions):
            for item, position in zip(data_pds, positions_pds):
                result[position] = item
        return result

    def _get_items_by_slice(self, start: int, stop: int, step: int) -> list[_T_co]:
        """Efficiently handle non-shuffled slice with arbitrary step."""
        slices = []

        cur_start = start
        cur_stop = stop
        ds_indices = []
        for idx, ds in enumerate(self.datasets):
            length = len(ds)
            if cur_start >= length:
                cur_start -= length
                cur_stop -= length
                continue

            ds_indices.append(idx)
            slices.append(slice(cur_start, min(cur_stop, length), step))

            if cur_stop <= length:
                break
            leftover = (length - cur_start) % step
            cur_start = 0 if leftover == 0 else step - leftover
            cur_stop -= length

        data = self._load(slices, ds_indices)
        return [item for sublist in data for item in sublist]

    def __len__(self) -> int:
        return self.length

    def release(self):
        '''Release parallel loading resources and dataset file handles.'''
        if self._loader is not None:
            self._loader.close()
        for ds in self.datasets:
            ds.release()


class Subset(Dataset[_T_co]):
    """
    Subset of a dataset with a given slice.
    
    If the dataset is a ConcatDataset, Subset will act on each sub-dataset of ConcatDataset and
    return a new ConcatDataset with all its sub-datasets sliced. This is to enable parallel loading
    using ConcatDataset, and its effect is completely equivalent to direct Subset.
    """

    dataset: Dataset[_T_co]

    @overload
    def __new__(cls, dataset: ConcatDataset, indices: Sequence[int]) -> ConcatDataset: ...
    @overload
    def __new__(cls, dataset: Dataset, indices: Sequence[int]) -> "Subset": ...
    @overload
    def __new__(cls, dataset: ConcatDataset, start: int, length: int) -> ConcatDataset:...
    @overload
    def __new__(cls, dataset: Dataset, start: int, length: int) -> "Subset": ...
    def __new__(cls, dataset, start_or_indices, length=None):
        if not isinstance(dataset, ConcatDataset):
            return super().__new__(cls)

        if cls._is_indices_mode(start_or_indices, length):
            return cls._distribute_indices(dataset, np.array(start_or_indices, dtype=int))
        return cls._distribute_interval(dataset, start_or_indices, length)

    @classmethod
    def _distribute_interval(cls, dataset: ConcatDataset, start: int, length: int):
        new_datasets = []
        for ds, ds_end_global in zip(dataset.datasets, dataset.cum_lengths):
            ds_start_global = ds_end_global - len(ds)

            # intersection of [start, stop) and [ds_start_global, ds_end_global)
            if start <= ds_start_global and start + length >= ds_end_global:
                new_datasets.append(ds)
                continue  # full overlap

            overlap_start = max(start, ds_start_global)
            overlap_end = min(start + length, ds_end_global)
            if overlap_start >= overlap_end:
                continue  # no overlap

            local_start = overlap_start - ds_start_global
            local_len = overlap_end - overlap_start
            new_datasets.append(cls(ds, local_start, local_len))

        return ConcatDataset(new_datasets)

    @classmethod
    def _distribute_indices(cls, dataset: ConcatDataset, indices: np.ndarray):
        new_datasets = []
        for ds, ds_end_global in zip(dataset.datasets, dataset.cum_lengths):
            ds_start_global = ds_end_global - len(ds)

            mask = (indices >= ds_start_global) & (indices < ds_end_global)
            local_idx = indices[mask] - ds_start_global
            if len(local_idx) == 0:
                continue
            new_datasets.append(cls(ds, local_idx))

        return ConcatDataset(new_datasets)

    def __init__(
        self, dataset: Dataset, start_or_indices: int | Sequence[int], length: int | None = None
    ):
        if self._is_indices_mode(start_or_indices, length):
            self.start = None
            self.length = len(start_or_indices)
            self.indices = np.array(start_or_indices, dtype=int)
        else:
            self.start = start_or_indices
            self.length = length
            self.indices = None
        self.dataset = dataset

    @staticmethod
    def _is_indices_mode(start_or_indices, length):
        if length is None:
            if not isinstance(start_or_indices, Sequence):
                raise TypeError("Arguments must be (start, length) or (indices,).")
            return True

        if not isinstance(start_or_indices, int) or not isinstance(length, int):
            raise TypeError("Arguments must be (start, length) or (indices,).")
        return False

    def __getitem__(self, index):
        if self.indices is not None:
            return self.dataset[self.indices[index]]

        if isinstance(index, int):
            index += self.start
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self)
            index = slice(start + self.start, stop + self.start, index.step)
        elif isinstance(index, (list, np.ndarray)):
            index = np.array(index, dtype=int) + self.start
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
        return self.dataset[index]

    def __len__(self):
        return self.length

    def release(self):
        '''Release dataset resources.'''
        self.dataset.release()
