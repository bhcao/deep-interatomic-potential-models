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

import logging

import pydantic

from dipm.data.chemical_datasets.utils import CHEMICAL_SYMBOLS

logger = logging.getLogger("dipm")


class DatasetInfo(pydantic.BaseModel):
    """Pydantic dataclass holding information computed from the dataset that is
    (potentially) required by the models. There are three types of fields:

    1. **User specified fields**: These fields are specified by the user but cannot be
       changed when fine-tuning.
    2. **Model related computed fields**: These fields are computed from the dataset but
       are bound to the model and cannot be changed when fine-tuning.
    3. **Dataset related computed fields**: These fields are computed from the dataset
       and can / are recommended to be changed when fine-tuning.

    Attributes:
        cutoff_distance_angstrom: The graph cutoff distance that was
                          used in the dataset in Angstrom.
        max_neighbors_per_atom: The maximum number of neighbors to consider for each atom.
                                Do NOT use it typically, as it will broke the smoothness.
        task_list: List of different tasks/datasets used in training. `None` (default)
                   means no task embedding used / only one task. If provided, values
                   of the atomic energies map must be lists of floats, one for each task.
        atomic_energies_map: A dictionary mapping the atomic numbers to the
                             computed average atomic energies for that element.
        avg_num_neighbors: The mean number of neighbors an atom has in the dataset.
        avg_num_nodes: The mean number of nodes per graph in the dataset.
        avg_r_min_angstrom: The mean minimum edge distance for a structure in the
                            dataset.
        scaling_mean: The mean used for the rescaling of the dataset values, the
                      default being 0.0.
        scaling_stdev: The standard deviation used for the rescaling of the dataset
                       values, the default being 1.0.
        median_num_neighbors: The median number of neighbors an atom has in the dataset.
        max_total_edges: The maximum number of edges in the dataset.
        median_num_nodes: The median number of nodes per graph in the dataset.
        max_num_nodes: The maximum number of nodes per graph in the dataset.
    """

    # User specified fields
    cutoff_distance_angstrom: float
    max_neighbors_per_atom: int | None = None
    task_list: list[str] | None = None

    # Model related computed fields
    atomic_energies_map: dict[int, float | list[float]]
    avg_num_neighbors: float = 1.0
    avg_num_nodes: float = 1.0
    avg_r_min_angstrom: float | None = None
    scaling_mean: float = 0.0
    scaling_stdev: float = 1.0

    # Dataset related computed fields
    median_num_neighbors: int = 1
    max_total_edges: int = 1
    median_num_nodes: int = 1
    max_num_nodes: int = 1

    @pydantic.field_validator("atomic_energies_map", mode="before")
    @classmethod
    def transform_atomic_energies_map(
        cls, value: dict[str | int, float | list[float | None]]
    ) -> dict[int, float]:
        """Transform string keys to int, None values to nan."""
        def none_to_nan(v):
            if isinstance(v, list):
                return [float('nan') if x is None else x for x in v]
            return v
        value = {int(k): none_to_nan(v) for k, v in value.items()}
        return value

    def __str__(self):
        atomic_energies_map_with_symbols = {
            CHEMICAL_SYMBOLS[num]: value for num, value in self.atomic_energies_map.items()
        }
        return (
            f"Atomic Energies: {atomic_energies_map_with_symbols}, "
            f"Task List: {self.task_list}, "
            f"Avg. num. neighbors: {self.avg_num_neighbors:.2f}, "
            f"Avg. num. nodes: {self.avg_num_nodes:.2f}, "
            f"Avg. r_min: {self.avg_r_min_angstrom:.2f}, "
            f"Graph cutoff distance: {self.cutoff_distance_angstrom}, "
            f"Max. num. neighbors: {self.max_neighbors_per_atom}, "
            f"Max. total edges: {self.max_total_edges}, "
            f"Median num. neighbors: {self.median_num_neighbors}, "
            f"Median num. nodes: {self.median_num_nodes}, "
        )
