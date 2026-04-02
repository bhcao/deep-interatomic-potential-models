# Copyright 2025 Zhongguancun Academy
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

from dipm.data.chemical_datasets.dataset import Dataset
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
        load_into_memory (bool): Whether to load the entire dataset into memory. Default is
            ``False``.
        use_shared_memory (bool): Whether to use shared memory to share the data between
            processes. Default is ``False``.
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
        load_into_memory: bool = False,
        use_shared_memory: bool = False,
    ):
        if num_workers is None:
            num_workers = mp.cpu_count()

        if num_workers < 0 or prefetch_factor < 0:
            raise ValueError("num_workers and prefetch_factor options should be non-negative.")

        if batch_size <= 0 or max_n_node <= 0 or max_n_edge <= 0:
            raise ValueError(
                "batch_size, max_n_node and max_n_edge must be positive integers."
            )

        if num_workers > 0:
            dataset = _ParallelDataset(dataset, num_workers, use_shared_memory)

        self.dataset = dataset
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.devices = devices
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.load_into_memory = load_into_memory
        self.use_shared_memory = use_shared_memory

        # Plus one for the extra padding node.
        self.n_node = batch_size * max_n_node + 1
        # Times two because we want backwards edges.
        self.n_edge = batch_size * max_n_edge * 2
        self.n_graph = batch_size + 1

        self._memory_cache = None
        if load_into_memory:
            cache = []
            for graph in self._sampler():
                cache.append(graph)
            self._memory_cache = cache

            self.dataset.release()
            return # Don't need to prefetch

        if prefetch_factor > 0:
            self.queue = queue.Queue(maxsize=prefetch_factor)

            # Start the prefetch
            self.thread = threading.Thread(target=self._prefetch, daemon=True)
            self.thread.start()

    def _sampler(self):
        """Generator that yields single shuffled graph from the dataset.

        We use batched getitem to reduce the overhead of opening file handles.
        """
        if self._memory_cache is not None:
            yield from self._memory_cache
            return

        dataset_len = len(self.dataset)
        num_devices = 1 if self.devices is None else len(self.devices)
        batch_size = (self.prefetch_factor // 3 + 1) * self.batch_size * num_devices

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

        # TODO: On some platforms, multiprocessing may lead to memory leaks. This is a temporary
        # solution to this problem.
        self.dataset.release()

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
            self._sampler(), self.n_node, self.n_edge, self.n_graph,
            skip_last_batch=self.drop_last, use_shared_memory=self.use_shared_memory
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
        if self.prefetch_factor > 0 and not self.load_into_memory:
            return self._prefetch_iter()
        return self._batch_sampler()


class _ParallelDataset(Dataset):
    """Helper class to load data in parallel."""

    def __init__(self, dataset: Dataset, num_workers: int, use_shared_memory: bool):
        dataset.release()
        self.dataset = dataset
        self.num_workers = num_workers
        self.use_shared_memory = use_shared_memory

        self.queues_in: list[mp.Queue] = []
        self.queue_out: mp.Queue = None
        self.processes: list[mp.Process] = []

        self.started = False

    @staticmethod
    def _worker(dataset: Dataset, queue_in: mp.Queue, queue_out: mp.Queue):
        while True:
            msg = queue_in.get()
    
            if msg is None:
                dataset.release()
                break
    
            task_id, indices = msg
    
            data = dataset[indices]
    
            queue_out.put((task_id, data))

    def _start_workers(self):
        ctx = mp.get_context("spawn")

        q_out = ctx.Queue()
        self.queue_out = q_out

        for _ in range(self.num_workers):
            q_in = ctx.Queue()

            p = ctx.Process(
                target=self._worker,
                args=(self.dataset, q_in, q_out),
            )
            p.start()

            self.queues_in.append(q_in)
            self.processes.append(p)

    def release(self):
        if not self.started:
            return

        for q in self.queues_in:
            q.put(None)

        for p in self.processes:
            p.join()

        for q in self.queues_in:
            q.close()
            q.join_thread()

        if self.queue_out is not None:
            self.queue_out.close()
            self.queue_out.join_thread()

        self.started = False

        self.processes.clear()
        self.queues_in.clear()
        self.queue_out = None

    def __len__(self):
        return len(self.dataset)

    def _normalize_index(self, index):
        # This class is only used by DataLoader, which won't create other types of indices.
        if isinstance(index, slice):
            return np.arange(*index.indices(len(self)))
        elif isinstance(index, np.ndarray):
            return index
        else:
            raise TypeError("ParallelDataset only supports slice or ndarray")

    def __getitem__(self, index):

        if not self.started:
            self._start_workers()
            self.started = True

        indices = self._normalize_index(index)

        splits = np.array_split(indices, self.num_workers)

        expected = 0
        for worker_id, split in enumerate(splits):
            if len(split) == 0:
                continue

            self.queues_in[worker_id].put((worker_id, split))
            expected += 1

        results = [None] * self.num_workers
        received = 0

        while received < expected:
            task_id, data = self.queue_out.get()

            results[task_id] = data
            received += 1

        merged = []
        for r in results:
            if r is None:
                continue
            merged.extend(r)

        if self.use_shared_memory:
            out = []
            for r in merged:
                out.append(r.to_graph())
            return out

        return merged

    def __del__(self):
        self.release()
