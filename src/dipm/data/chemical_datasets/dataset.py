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

from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging
import multiprocessing as mp
from typing import overload

import numpy as np

from dipm.data.chemical_system import ChemicalSystem

DEFAULT_WEIGHT = 1.0
DEFAULT_PBC = np.zeros(3, bool)
DEFAULT_CELL = np.zeros((3, 3))

logger = logging.getLogger('dipm')


class Dataset(ABC):
    """Pytorch dataset like base objects."""

    @overload
    def __getitem__(self, index: int) -> ChemicalSystem:...
    @overload
    def __getitem__(self, index: list | np.ndarray | slice) -> list[ChemicalSystem]:...
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")

    @abstractmethod
    def set_post_process_fn(self, fn):
        """Set a post-processing function to be applied to each system before returning it. Return
        None to filter out the system."""
        raise NotImplementedError("Subclasses of Dataset should implement set_post_process_fn.")

    def release(self):
        """Release resources."""

    def __del__(self):
        self.release()


class _ParallelLoader:
    """Helper class to load data in parallel."""

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.queues_in = []
        self.queues_out = []
        self.process_pool = []

    def _worker(self, dataset: Dataset, queue_in: mp.Queue, queue_out: mp.Queue):
        """Worker function to load data from a dataset."""
        while True:
            index = queue_in.get()
            if index is None:
                break
            queue_out.put(dataset[index])

    def start(self):
        '''Start the worker processes.'''
        for ds in self.datasets:
            queue_in = mp.Queue()
            queue_out = mp.Queue()
            self.queues_in.append(queue_in)
            self.queues_out.append(queue_out)
            process = mp.Process(target=self._worker, args=(ds, queue_in, queue_out))
            process.start()
            self.process_pool.append(process)

    def stop(self):
        '''Stop the worker processes.'''
        for queue_in in self.queues_in:
            queue_in.put(None)

        # Wait for all processes to finish
        for p in self.process_pool:
            p.join()

        self.process_pool.clear()

    def load(
        self,
        indices: int | list | np.ndarray | slice,
        ds_indices: int | list[int]
    ) -> ChemicalSystem | list[list[ChemicalSystem]]:
        """Load data from the datasets in parallel."""

        # single index, file handler is holded by the worker process
        if isinstance(indices, int):
            self.queues_in[ds_indices].put(indices)
            return self.queues_out[ds_indices].get()

        # batch indices
        for idx, ds_idx in zip(indices, ds_indices):
            self.queues_in[ds_idx].put(idx)

        results = []
        for ds_idx in ds_indices:
            data = self.queues_out[ds_idx].get()
            results.append(data)

        return results


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets. Pytorch-like ConcatDataset."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        shuffle: bool = False,
        parallel: bool = False,
    ):
        """
        Create a concatenated dataset.

        Args:
            datasets (sequence): List of datasets to concatenate.
            shuffle (bool): Whether to shuffle all datasets together.
            parallel (bool): Whether to load data in parallel. If True, every dataset will be
                loaded in a separate process when `__getitem__` is called.
        """
        self.datasets = datasets
        self.lengths = np.array([len(d) for d in self.datasets])
        self.cum_lengths = np.cumsum(self.lengths)
        self.length = int(self.cum_lengths[-1])

        self.shuffle = shuffle
        if self.shuffle:
            self.indices = np.random.permutation(self.length)

        # parallel loading
        self.parallel = parallel
        self._loader = None

    def _load(self, indices, ds_indices):
        """Load data from the datasets in parallel or sequentially."""
        if self.parallel:
            if self._loader is None:
                self._loader = _ParallelLoader(self.datasets)
                self._loader.start()
            return self._loader.load(indices, ds_indices)

        # sequential
        if isinstance(indices, int):
            item = self.datasets[ds_indices][indices]
            return item

        results = []
        for idx, ds_idx in zip(indices, ds_indices):
            data = self.datasets[ds_idx][idx]
            results.append(data)
        return results

    def __getitem__(self, index):
        if self.shuffle:
            index = self.indices[index]

        if isinstance(index, slice):
            return self._get_items_by_slice(
                index.start or 0, index.stop or len(self), index.step or 1
            )

        if isinstance(index, (list, np.ndarray)):
            indices = np.array(index, dtype=int)
            return self._get_items_by_indices(indices)

        return self._get_item(index)

    def _get_item(self, index: int) -> ChemicalSystem:
        '''Get one item by index.'''
        dataset_idx = np.searchsorted(self.cum_lengths, index, side="right")
        if dataset_idx == 0:
            local_idx = index
        else:
            local_idx = index - self.cum_lengths[dataset_idx - 1]
        return self._load(local_idx, dataset_idx)

    def _get_items_by_indices(self, indices: np.ndarray) -> list[ChemicalSystem]:
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

    def _get_items_by_slice(self, start: int, stop: int, step: int) -> list[ChemicalSystem]:
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

    def set_post_process_fn(self, fn):
        """Recursively set post-processing function to all datasets."""
        for ds in self.datasets:
            ds.set_post_process_fn(fn)

    def release(self):
        '''Release parallel loading resources and dataset file handles.'''
        if self._loader is not None:
            self._loader.stop()
        for ds in self.datasets:
            ds.release()


class Subset(Dataset):
    """
    Subset of a dataset with a given slice.
    
    If the dataset is a ConcatDataset, Subset will act on each sub-dataset of ConcatDataset and
    return a new ConcatDataset with all its sub-datasets sliced. This is to enable parallel loading
    using ConcatDataset, and its effect is completely equivalent to direct Subset.
    """

    @overload
    def __new__(cls, dataset: ConcatDataset, start: int, length: int) -> ConcatDataset:...
    @overload
    def __new__(cls, dataset: Dataset, start: int, length: int) -> "Subset": ...
    def __new__(cls, dataset: Dataset, start: int, length: int):
        if not isinstance(dataset, ConcatDataset):
            return super().__new__(cls)

        # From a probabilistic perspective, almost all datasets will be kept.
        if dataset.shuffle:
            concat_ds = ConcatDataset(
                dataset.datasets,
                parallel=dataset.parallel,
            )
            concat_ds.shuffle = True
            concat_ds.indices = dataset.indices[start:start+length]
            concat_ds.length = length
            return concat_ds

        prev_cum = 0
        new_datasets = []
        for ds, ds_end_global in zip(dataset.datasets, dataset.cum_lengths):
            ds_start_global = prev_cum
            prev_cum = ds_end_global

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

        return ConcatDataset(
            new_datasets,
            parallel=dataset.parallel,
        )

    def __init__(self, dataset: Dataset, start: int, length: int):
        self.dataset = dataset
        self.start = start
        self.length = length

    def __getitem__(self, index):
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

    def set_post_process_fn(self, fn):
        """Set post-processing function to the dataset."""
        self.dataset.set_post_process_fn(fn)

    def release(self):
        '''Release dataset resources.'''
        self.dataset.release()
