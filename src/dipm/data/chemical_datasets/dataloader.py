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

from collections.abc import Generator
import functools
import logging
import multiprocessing as mp
import queue
import threading

import jax
import jraph
import numpy as np

from dipm.data.chemical_datasets.dataset import Dataset, ConcatDataset
from dipm.data.helpers.dynamically_batch import dynamically_batch

logger = logging.getLogger('dipm')

_GraphDataset = Dataset[jraph.GraphsTuple]


class DataLoader:
    r"""Pytorch-like data loader that provides an iterable which returns ``jraph.GraphsTuple`` over
    the given dataset.

    Args:
        dataset (Dataset[GraphTuple]): Dataset from which to load the data. Must return
            ``jraph.GraphsTuple`` objects.
        batch_size (int): How many samples per batch to load.
        max_n_node (int): The maximum number of nodes contributed by one graph in a batch.
        max_n_edge (int): The maximum number of edges contributed by one graph in a batch.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        devices (int): Jax devices to put tensors on. None means that no sharding will
            be done.
        shuffle (bool): Whether to have the data reshuffled at every epoch. If ``False``, may
            causing unbalanced load among workers. Default is ``True``.
        num_workers (int): How many subprocesses to use for data loading. ``0`` means that
            the data will be loaded in the main process. Only ``ConcatDataset`` is supported
            to use multiple workers. The number of processes will always be less than or equal
            to the number of subdatasets. If ``None``, will use ``min(num_files, num_cpus)``
            subprocesses.
        prefetch_factor (int): Number of batches to load in advance. The amount of data
            loaded at once is one third of this number. Default is ``128`` to reduce the
            overhead of opening file handles. A additional thread is used to prefetch the data.
    """

    def __init__(
        self,
        dataset: _GraphDataset,
        batch_size: int,
        max_n_node: int,
        max_n_edge: int,
        drop_last: bool = False,
        devices: list[jax.Device] | None = None, # type: ignore
        shuffle: bool = True,
        num_workers: int | None = None,
        prefetch_factor: int = 128,
    ):
        if num_workers is None:
            if isinstance(dataset, ConcatDataset):
                num_workers = min(mp.cpu_count(), len(dataset.datasets))
            else:
                num_workers = 0

        if num_workers < 0 or prefetch_factor < 0:
            raise ValueError("num_workers and prefetch_factor options should be non-negative.")

        if batch_size <= 0 or max_n_node <= 0 or max_n_edge <= 0:
            raise ValueError(
                "batch_size, max_n_node and max_n_edge must be positive integers."
            )

        if num_workers > 0:
            if isinstance(dataset, ConcatDataset):
                if num_workers > len(dataset.datasets):
                    num_workers = len(dataset.datasets)
                    logger.warning(
                        "num_workers is set to %s because there are only %s subdatasets.",
                        num_workers, len(dataset.datasets)
                    )
                dataset._loader = _ParallelLoader(dataset.datasets, num_workers)
            else:
                num_workers = 0
                logger.warning("num_workers is disabled because dataset is not ConcatDataset.")

        self.dataset = dataset
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.devices = devices
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        # Plus one for the extra padding node.
        self.n_node = batch_size * max_n_node + 1
        # Times two because we want backwards edges.
        self.n_edge = batch_size * max_n_edge * 2
        self.n_graph = batch_size + 1

        if prefetch_factor > 0:
            self.queue = queue.Queue(maxsize=prefetch_factor)

            # Start the prefetch
            self.thread = threading.Thread(target=self._prefetch, daemon=True)
            self.thread.start()

    def _sampler(self):
        """Generator that yields single shuffled graph from the dataset.

        We use batched getitem to reduce the overhead of opening file handles.
        """
        dataset_len = len(self.dataset)
        num_devices = 1 if self.devices is None else len(self.devices)
        batch_size = self.prefetch_factor * self.batch_size * num_devices // 3

        if self.shuffle:
            indices = np.random.permutation(dataset_len)
            def shuffle_fn(x):
                return indices[x]
        else:
            def shuffle_fn(x):
                return x

        for i in range(0, dataset_len, batch_size):
            idx = shuffle_fn(slice(i, min(i + batch_size, dataset_len)))
            yield from self.dataset[idx]

    def _parallel_accumulate(self, generator):
        """Accumulate the graphs for parallel training."""
        num_devices = len(self.devices)
        device_shard_fn = functools.partial(
            jax.tree.map,
            lambda *x: jax.device_put_sharded(x, self.devices),
        )

        batch = []
        for i, graph in enumerate(generator):
            if i % num_devices == num_devices - 1:
                batch.append(graph)
                # pylint: disable=no-value-for-parameter
                yield device_shard_fn(*batch)
                batch = []
            else:
                batch.append(graph)

    def _batch_sampler(self):
        """Generator that yields batched and parallel graphs from the dataset."""
        batch_sampler = dynamically_batch(
            self._sampler(), self.n_node, self.n_edge, self.n_graph, skip_last_batch=self.drop_last
        )

        if self.devices is not None:
            batch_sampler = self._parallel_accumulate(batch_sampler)

        return batch_sampler

    def _prefetch(self):
        """Prefetch items from the original iterable into the queue.

        This method also adds a None at the end to indicate the end of the iterator.
        """
        for item in self._batch_sampler():
            self.queue.put(item)  # This will block when the queue is full

        # Indicate the end of the iterator
        self.queue.put(None)

    def _prefetch_iter(self):
        """Implementation of the prefetch iterator. It starts a new thread once completed."""
        item = self.queue.get()
        while item is not None:
            yield item
            item = self.queue.get()

        # Restart a new prefetch cycle
        assert not self.thread.is_alive()  # it should be dead
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def __iter__(self) -> Generator[jraph.GraphsTuple, None, None]:
        """Returns an iterator over the dataset."""
        if self.prefetch_factor > 0:
            return self._prefetch_iter()
        return self._batch_sampler()

    def __del__(self):
        if isinstance(self.dataset, ConcatDataset) and self.dataset._loader is not None:
            del self.dataset._loader
            self.dataset._loader = None


class _ParallelLoader:
    r"""Helper class to load data from ConcatDataset in parallel using multiprocessing.Pool."""

    def __init__(self, datasets: list[_GraphDataset], num_workers: int):
        assert len(datasets) > 0 and num_workers > 0, "No need to use parallel loader."
        self.datasets = datasets
        self.num_workers = num_workers
        self.pool = None

    @staticmethod
    def _worker(args):
        dataset, index = args
        return dataset[index]

    def close(self):
        """Close the pool and join the processes."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __call__(
        self, indices: list[tuple[int, list | np.ndarray]]
    ) -> list[list[jraph.GraphsTuple]]:
        """Load data from the datasets in parallel.
        
        Args:
            indices: A list of dataset indices and indices to load from each dataset.
        """

        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(self.num_workers)

        for d in self.datasets:
            d.release()

        args = [(self.datasets[ds], idx) for ds, idx in indices]
        return self.pool.map(self._worker, args)

    def __del__(self):
        self.close()
