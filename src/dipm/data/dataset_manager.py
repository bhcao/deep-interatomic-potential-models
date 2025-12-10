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

import functools
import logging
import multiprocessing as mp

import jax
import jraph
import numpy as np
from tqdm import tqdm

from dipm.data.chemical_datasets.dataloader import DataLoader
from dipm.data.chemical_datasets.dataset_creation import DatasetCreation
from dipm.data.chemical_datasets.utils import CHEMICAL_SYMBOLS
from dipm.data.chemical_system import ChemicalSystem
from dipm.data.dataset_info import DatasetInfo
from dipm.data.chemical_datasets.dataset import Dataset
from dipm.data.chemical_datasets.utils import filter_broken_system_and_get_stats
from dipm.data.helpers.atomic_number_table import AtomicNumberTable
from dipm.data.helpers.compute_dataset_info import MedianHistogram, compute_dataset_info_from_stats
from dipm.data.configs import DatasetManagerConfig
from dipm.data.helpers.graph_creation import create_graph_from_chemical_system

logger = logging.getLogger("dipm")


def _process_worker(
    dataset: Dataset, queue: mp.Queue, calc_stats=False, calc_part=False, calc_spices=False
):
    '''Worker function to process a dataset in parallel.'''
    data = []
    exclude_ids = []

    if calc_part or calc_stats:
        neighbors_hist = MedianHistogram(init_bins=512)
        for i, d in enumerate(dataset):
            if d is None:
                exclude_ids.append(i)
            else:
                neighbors_hist.add(d.pop('num_neighbors').astype(np.int64))
                data.append(d)
            if i % 100 == 0:
                queue.put(100)
    else:
        for i, d in enumerate(dataset):
            if d is None:
                exclude_ids.append(i)
            else:
                data.append(d)
            if i % 100 == 0:
                queue.put(100)

    dataset.release()

    if len(data) % 100 != 0:
        queue.put(len(data) % 100)

    exclude_path = dataset.path.with_name(dataset.path.stem + '_exclude.npy')
    exclude_path.unlink(missing_ok=True)

    warning = None
    if len(exclude_ids) > 0:
        warning = f"Discarded {len(exclude_ids)} graphs due to having no edges or unseen elements."
        with open(exclude_path, 'wb') as f:
            np.save(f, np.array(exclude_ids, dtype=int))

    output = {}

    if calc_part or calc_stats:
        output.update({
            'max_total_edges': np.max([d['total_edges'] for d in data]),
            'num_nodes': np.stack([d['num_nodes'] for d in data]),
            'num_neighbors_hist': neighbors_hist,
        })

    if calc_spices or calc_stats:
        species = np.unique(np.concatenate([d['species'] for d in data]))
        output['species'] = species

    if calc_stats:
        species_indices = {s: i for i, s in enumerate(species)}
        species_count = np.zeros((len(data), len(species)), dtype=int)
        for i, d in enumerate(data):
            for s, c in zip(d['species'], d['species_count']):
                species_count[i, species_indices[s]] = c
        output.update({
            'min_neighbor_distance': np.sum([d['min_neighbor_distance'] for d in data]),
            'species_count': species_count,
            'energy': np.stack([d['energy'] for d in data]),
            'num_graphs': len(data),
            'task': dataset.task,
        })

    return output, warning


def _process_datasets_in_parallel(
    datasets: list[Dataset], calc_stats=False, calc_part=False, calc_spices=False, desc=""
):
    manager = mp.Manager()
    queue = manager.Queue()
    total_steps = sum(len(ds) for ds in datasets)

    with mp.get_context("spawn").Pool(processes=min(len(datasets), mp.cpu_count())) as pool:
        results = [
            pool.apply_async(_process_worker, args=(
                dataset, queue, calc_stats, calc_part, calc_spices
            ))
            for dataset in datasets
        ]

        steps = 0
        with tqdm(total=total_steps, desc=desc) as pbar:
            while steps < total_steps:
                step = queue.get()
                pbar.update(step)
                steps += step

        outputs = []
        for r in results:
            output, warning = r.get()
            if warning is not None:
                logger.warning(warning)
            outputs.append(output)

    return outputs


class _CreateGraphFn:
    """Pickleable callable graph creator."""

    def __init__(
        self,
        dataset_info: DatasetInfo,
        use_formation_energies: bool,
        n_node: int,
        n_edge: int,
    ):
        self._info = dataset_info
        self.z_table = AtomicNumberTable(sorted(self._info.atomic_energies_map.keys()))
        self.to_index_fun = np.vectorize(self.z_table.z_to_index)
        self.use_formation_energies = use_formation_energies
        self.n_node = n_node
        self.n_edge = n_edge

    def __call__(self, chemical_system: ChemicalSystem) -> jraph.GraphsTuple:
        chemical_system.atomic_species = self.to_index_fun(chemical_system.atomic_numbers)

        if self.use_formation_energies:
            chemical_system.energy = self._to_formation_energy(
                chemical_system.energy,
                chemical_system.atomic_species,
                chemical_system.task,
            )

        graph = create_graph_from_chemical_system(
            chemical_system, self._info.cutoff_distance_angstrom, self._info.max_neighbors_per_atom
        )
        self._check_graph(graph)
        return graph

    def _check_graph(self, graph: jraph.GraphsTuple):
        """Check if a graph is valid."""
        if graph.n_node.item() == 0:
            raise ValueError("Graph has no nodes.")
        if graph.n_edge.item() == 0:
            raise ValueError("Graph has no edges.")
        if graph.n_node.item() >= self.n_node:
            raise ValueError(f"Graph has more than {self.n_node - 1} nodes.")
        if graph.n_edge.item() > self.n_edge:
            raise ValueError(f"Graph has more than {self.n_edge} edges.")

    def _get_energy(self, key, task):
        z = self.z_table.index_to_z(key)
        value = self._info.atomic_energies_map.get(z, None)
        if value is None:
            return 0.0
        return value[task] if self._info.task_list is not None else value

    def _to_formation_energy(self, energy, species, task):
        sum_atomic = sum(self._get_energy(k, task) for k in species)
        return energy - sum_atomic


class DatasetManager:
    """Prepares a dataset by calculating statistics and storing masks. This is a part of the 
    `DatasetManager` class.
    
    Args:
        config (DatasetManagerConfig): Configuration object containing the dataset paths etc.
    """
    Config = DatasetManagerConfig

    def __init__(self, config: DatasetManagerConfig, dataset_creation: DatasetCreation):
        self.config = config
        self.ds_create = dataset_creation
        self.dataset_info = None

    def _check_args_consistency(self, dataset_info: DatasetInfo):
        if self.config.graph_cutoff_angstrom != dataset_info.cutoff_distance_angstrom:
            raise ValueError(
                "DatasetManager got inconsistent cutoff distance: "
                "pass `None` as dataset_info to create a fresh dataset, or fix "
                "dataset_config if you want to reuse the dataset_info."
            )
        if self.config.max_neighbors_per_atom != dataset_info.max_neighbors_per_atom:
            raise ValueError(
                "DatasetManager got inconsistent maximum number of neighbors per atom: "
                "pass `None` as dataset_info to create a fresh dataset, or fix "
                "dataset_config if you want to reuse the dataset_info."
            )
        if self.ds_create.task_list != dataset_info.task_list:
            raise ValueError(
                "DatasetManager got inconsistent task list: "
                "pass `None` as dataset_info to create a fresh dataset, or fix "
                "dataset_config if you want to reuse the dataset_info."
            )

        if not self.config.update_dataset_info:
            if self.config.max_n_node is not None:
                max_num_nodes = self.config.max_n_node * self.config.batch_size
                if max_num_nodes < dataset_info.max_num_nodes:
                    raise ValueError(
                        "max_num_nodes in config is smaller than the one in dataset_info."
                        "Please set update_dataset_info to True to update the dataset_info."
                    )

            if self.config.max_n_edge is not None:
                max_total_edges = self.config.max_n_edge * self.config.batch_size
                if max_total_edges < dataset_info.max_total_edges:
                    raise ValueError(
                        "max_total_edges in config is smaller than the one in dataset_info."
                        "Please set update_dataset_info to True to update the dataset_info."
                    )

    def prepare_datasets(self, dataset_info: DatasetInfo | None = None) -> DatasetInfo:
        """Calculate statistics and store masks. Shuffling and splitting is disabled.

        Args:
            dataset_info (DatasetInfo, optional): The dataset information. If provided, 
                only to remove unseen elements and empty graphs.

        Returns:
            The calculated/updated dataset information.
        """
        # Sanity check when DatasetInfo is passed from the outside
        if dataset_info is not None:
            self._check_args_consistency(dataset_info)

        _cfg = self.config

        calc_stats = dataset_info is None
        calc_spices = _cfg.drop_unseen_elements
        calc_part = _cfg.update_dataset_info
        cutoff = _cfg.graph_cutoff_angstrom
        max_neighbors = _cfg.max_neighbors_per_atom
        max_total_edges = None if _cfg.max_n_edge is None else _cfg.max_n_edge * _cfg.batch_size
        max_num_nodes = None if _cfg.max_n_node is None else _cfg.max_n_node * _cfg.batch_size

        if not (calc_stats or calc_spices or calc_part):
            logger.info("Dataset preparation skipped.")
            self.dataset_info = dataset_info
            return dataset_info

        logger.info("Preparing datasets...")

        max_set = None if calc_stats else set(dataset_info.atomic_energies_map.keys())

        train_filter_fn = functools.partial(
            filter_broken_system_and_get_stats,
            cutoff=cutoff,
            max_neighbours=max_neighbors,
            max_set=max_set,
            max_num_nodes=max_num_nodes,
            max_total_edges=max_total_edges,
            calc_stats=calc_stats,
            calc_part=calc_part,
            calc_spices=calc_spices,
        )

        train_datasets = self.ds_create.get_train_datasets(train_filter_fn, load_exclusions=False)

        outputs = _process_datasets_in_parallel(
            train_datasets, calc_stats, calc_part, calc_spices, desc="Analyzing training set"
        )

        if dataset_info is None:
            dataset_info = compute_dataset_info_from_stats(
                outputs, cutoff, max_neighbors, self.ds_create.task_list
            )
        else:
            if calc_part:
                dataset_info = compute_dataset_info_from_stats(
                    outputs, cutoff, max_neighbors, self.ds_create.task_list, dataset_info
                )
            if calc_spices:
                species = np.unique(np.concatenate([d['species'] for d in outputs]))
                dropped_species = set(dataset_info.atomic_energies_map.keys()) - set(species)
                dataset_info.atomic_energies_map = {
                    s: dataset_info.atomic_energies_map[s] for s in species
                }
                if len(dropped_species) > 0:
                    logger.info(
                        "Elements %s are not seen in the training set and will be dropped.",
                        ", ".join([CHEMICAL_SYMBOLS[i] for i in dropped_species])
                    )

        max_set = set(dataset_info.atomic_energies_map.keys())
        valid_test_filter_fn = functools.partial(
            filter_broken_system_and_get_stats,
            cutoff=cutoff,
            max_set=max_set,
            max_num_nodes=max_num_nodes,
            max_total_edges=max_total_edges,
        )

        valid_datasets = self.ds_create.filter_duplicates(
            self.ds_create.get_valid_datasets(valid_test_filter_fn, load_exclusions=False),
            train_datasets,
        )
        test_datasets = self.ds_create.filter_duplicates(
            self.ds_create.get_test_datasets(valid_test_filter_fn, load_exclusions=False),
            train_datasets + valid_datasets,
        )

        _process_datasets_in_parallel(
            valid_datasets + test_datasets, desc="Analyzing validation/test set"
        )
        logger.info("Datasets preparation completed.")
        self.dataset_info = dataset_info

        return dataset_info

    def get_loaders(
        self, devices: list[jax.Device] | None = None # type: ignore
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """Get data loaders. Shuffling and splitting is enabled.

        Args:
            devices (list[jax.Device], optional): The devices to use for training. If provided,
                parallel training is enabled.

        Returns:
            A tuple of training, validation (if provided), and test (if provided) dataloaders.
        """

        if self.dataset_info is None:
            raise ValueError("DatasetManager.prepare_datasets() must be called first.")

        max_n_node, max_n_edge = self._determine_autofill_batch_limitations()

        # Used for checking the validity of the graph
        n_node = self.config.batch_size * max_n_node + 1
        n_edge = self.config.batch_size * max_n_edge * 2
        train_dataset, valid_dataset, test_dataset = self.ds_create.create_datasets(
            _CreateGraphFn(self.dataset_info, self.config.use_formation_energies, n_node, n_edge),
        )

        create_dataloader_fn = functools.partial(
            DataLoader,
            batch_size=self.config.batch_size,
            max_n_node=max_n_node,
            max_n_edge=max_n_edge,
            devices=devices,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.num_batch_prefetch,
        )

        train_loader = create_dataloader_fn(train_dataset)
        if valid_dataset is not None:
            valid_loader = create_dataloader_fn(valid_dataset)
        else:
            valid_loader = None
        if test_dataset is not None:
            test_loader = create_dataloader_fn(test_dataset)
        else:
            test_loader = None

        return train_loader, valid_loader, test_loader

    def _determine_autofill_batch_limitations(self) -> tuple[int, int]:
        _cfg = self.config
        _info = self.dataset_info

        # Autofill max_n_node and max_n_edge if they are set to None
        if _cfg.max_n_node is None:
            max_n_node = _info.median_num_nodes
            if _cfg.batch_size * max_n_node < _info.max_num_nodes:
                logger.debug("Largest graph does not fit into batch -> resizing it.")
                max_n_node = int(np.ceil(_info.max_num_nodes / _cfg.batch_size))

            logger.debug(
                "The batching parameter max_n_node has been computed to be %s.",
                max_n_node,
            )
        else:
            max_n_node = _cfg.max_n_node
            if _cfg.batch_size * max_n_node < _info.max_num_nodes:
                raise ValueError(
                    "Largest graph does not fit into batch. Please increase max_n_node or "
                    "batch_size, or re-prepare your dataset with a smaller max_num_nodes "
                    "to filter out larger graphs."
                )

        if _cfg.max_n_edge is None:
            max_n_edge = _info.median_num_neighbors * max_n_node // 2

            if max_n_edge * _cfg.batch_size < _info.max_total_edges:
                logger.debug("Largest graph does not fit into batch -> resizing it.")
                max_n_edge = int(np.ceil(_info.max_total_edges / _cfg.batch_size))

            logger.debug(
                "The batching parameter max_n_edge has been computed to be %s.",
                max_n_edge,
            )
        else:
            max_n_edge = _cfg.max_n_edge
            if max_n_edge * _cfg.batch_size < _info.max_total_edges:
                raise ValueError(
                    "Largest graph does not fit into batch. Please increase max_n_edge or "
                    "batch_size, or re-prepare your dataset with a smaller max_total_edges "
                    "to filter out larger graphs."
                )

        return max_n_node, max_n_edge
