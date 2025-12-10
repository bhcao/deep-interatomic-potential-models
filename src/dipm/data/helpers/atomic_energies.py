# Copyright 2025 InstaDeep Ltd
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
#
# Modifications Copyright 2025 Cao Bohan
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

from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("dipm")


def _compute_average_e0s_from_task_statistics(
    species: list[np.ndarray], # [n_files, n_species]
    species_count: list[np.ndarray], # [n_files, n_graphs, n_species]
    energies: list[np.ndarray], # [n_files, n_graphs]
    task_name: str | None,
) -> dict[int, float]:
    """Compute average energy contribution of each element by least squares for
    a given task.

    Args:
        species: The list of species appearing for each file.
        species_count: The number of occurrences of each species for each graph and file.
        energies: The list of energies for each graph and file.
        task_name: The name of the task.

    Returns:
        The atomic energies dictionary which is the mapping of atomic species to
        the average energy contribution of each element.
    """

    num_graphs = sum(len(s) for s in energies)
    unique_species = np.unique(np.concatenate(species))

    species_count_concat = np.zeros((num_graphs, len(unique_species)), dtype=int)

    start = 0
    for species_i, count_i in zip(species, species_count):
        end = start + len(count_i)
        indices = np.searchsorted(unique_species, species_i)
        species_count_concat[start:end, indices] = count_i
        start = end

    energies = np.concatenate(energies)

    try:
        e0s_t = np.linalg.lstsq(species_count_concat, energies, rcond=1e-8)[0]
        atomic_energies = {s: e0s_t[i] for i, s in enumerate(unique_species)}

    except np.linalg.LinAlgError:
        logger.warning(
            "Failed to compute E0s %susing "
            "least squares regression, using the 0.0 for all atoms.",
            f"for task {task_name} " if task_name is not None else "",
        )
        atomic_energies = dict.fromkeys(unique_species, 0.0)

    return atomic_energies


def compute_average_e0s_from_statistics(
    species: list[np.ndarray], # [n_files, n_species]
    species_count: list[np.ndarray], # [n_files, n_graphs, n_species]
    energies: list[np.ndarray], # [n_files, n_graphs]
    tasks: list[np.ndarray], # [n_files,]
    task_list: list[str] | None = None,
) -> dict[int, float | list[float]]:
    """Compute average energy contribution of each element by least squares.

    Args:
        species: The list of species appearing for each file.
        species_count: The number of occurrences of each species for each graph and file.
        energies: The list of energies for each graph and file.
        tasks: The list of tasks for each file.
        task_list: The names of different tasks/datasets.

    Returns:
        The atomic energies dictionary which is the mapping of atomic species to
        the average energy contribution of each element.
    """

    species_task = defaultdict(list)
    species_count_task = defaultdict(list)
    energy_task = defaultdict(list)
    for s, c, e, t in zip(species, species_count, energies, tasks):
        species_task[t].append(s)
        species_count_task[t].append(c)
        energy_task[t].append(e)

    if task_list is None:
        assert set(species_task.keys()) == {None}
        return _compute_average_e0s_from_task_statistics(
            species_task[None],
            species_count_task[None],
            energy_task[None],
            None,
        )

    assert set(species_task.keys()) == set(range(len(task_list)))

    unique_species = np.unique(np.concatenate(species))
    atomic_energies = {species_number: [] for species_number in unique_species}

    for task, name in zip(species_task, task_list):
        atomic_energies_t = _compute_average_e0s_from_task_statistics(
            species_task[task],
            species_count_task[task],
            energy_task[task],
            name,
        )
        for atomic_number in atomic_energies:
            atomic_energies[atomic_number].append(
                atomic_energies_t.get(atomic_number, np.nan)
            )

    return atomic_energies
