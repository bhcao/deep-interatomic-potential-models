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

import numpy as np

from dipm.data.helpers.atomic_energies import compute_average_e0s_from_statistics
from dipm.data.dataset_info import DatasetInfo


class MedianHistogram:
    '''A class to manage and compute the median from a histogram.'''

    def __init__(self, init_bins=512):
        self.size = init_bins
        self.hist = np.zeros(self.size, dtype=np.int64)
        self.total_count = 0

    def _grow(self, new_max_value):
        """Grow histogram until it can hold new_max_value."""
        new_size = self.size
        while new_size <= new_max_value:
            new_size *= 2

        new_hist = np.zeros(new_size, dtype=np.int64)
        new_hist[:self.size] = self.hist
        self.hist = new_hist
        self.size = new_size

    def add(self, values):
        """Add a list of values."""
        max_v = int(values.max())
        if max_v >= self.size:
            self._grow(max_v)

        cnt = np.bincount(values, minlength=self.size)
        self.hist[:len(cnt)] += cnt
        self.total_count += len(values)

    @staticmethod
    def merge(others: 'list[MedianHistogram]') -> 'MedianHistogram':
        """Merge multiple histograms into one."""
        max_hist_size = max(len(other.hist) for other in others)
        merged_hist = MedianHistogram(max_hist_size)

        for other in others:
            merged_hist.total_count += other.total_count
            merged_hist.hist[:len(other.hist)] += other.hist

        return merged_hist

    def median(self):
        """Compute the median from the histogram."""
        midpoint = (self.total_count - 1) // 2
        cumulative = 0
        for idx, count in enumerate(self.hist):
            cumulative += count
            if cumulative > midpoint:
                return idx
        return len(self.hist) - 1

    def sum(self):
        """Return the total value of the histogram."""
        coeffs = np.arange(len(self.hist))
        return np.dot(self.hist, coeffs)


def compute_dataset_info_from_stats(
    stats: list[dict],
    cutoff_distance_angstrom: float,
    max_neighbors: int | None = None,
    task_list: list[str] | None = None,
    dataset_info: DatasetInfo | None = None,
) -> DatasetInfo:
    """Computes the dataset info from statistics.

    Fields must include: 'max_total_edges', 'num_nodes', 'num_neighbors_hist'.
    Optional when dataset_info is provided: 'min_neighbor_distance', 'species',
    'species_count', 'energy', 'num_graphs', 'task'.

    Args:
        stats: A list of dictionaries containing the statistics for each file.
        cutoff_distance_angstrom: The graph distance cutoff in Angstrom to
                                  store in the dataset info.
        max_neighbors: The maximum number of neighbors to consider for each atom.
        task_list: List of different tasks/datasets.
        dataset_info: An optional dataset info object to update. If specified,
                      only the dataset related computed fields will be updated.

    Returns:
        The dataset info object populated with the computed data.
    """
    num_nodes = np.concatenate([d['num_nodes'] for d in stats])
    neighbors_hist = MedianHistogram.merge([d['num_neighbors_hist'] for d in stats])

    if dataset_info is None:
        total_nodes = np.sum(num_nodes)
        num_graphs = np.sum([d['num_graphs'] for d in stats])

        avg_min_neighbor_distance = (
            np.sum([d['min_neighbor_distance'] for d in stats]) / num_graphs
        )
        atomic_energies_map = compute_average_e0s_from_statistics(
            [d['species'] for d in stats],
            [d['species_count'] for d in stats],
            [d['energy'] for d in stats],
            [d['task'] for d in stats],
            task_list=task_list,
        )

        dataset_info = DatasetInfo(
            # User specified fields
            cutoff_distance_angstrom=cutoff_distance_angstrom,
            max_neighbors_per_atom=max_neighbors,
            task_list=task_list,
            # Model related computed fields
            atomic_energies_map=atomic_energies_map,
            avg_num_neighbors=neighbors_hist.sum() / total_nodes,
            avg_num_nodes=total_nodes / num_graphs,
            avg_r_min_angstrom=avg_min_neighbor_distance,
            scaling_mean=0.0,
            scaling_stdev=1.0,
        )

    # Dataset related computed fields
    dataset_info.median_num_neighbors = int(neighbors_hist.median())
    dataset_info.max_total_edges = int(np.max([d['max_total_edges'] for d in stats])) // 2
    dataset_info.median_num_nodes = int(np.ceil(np.median(num_nodes)))
    dataset_info.max_num_nodes = int(np.max(num_nodes))

    return dataset_info
