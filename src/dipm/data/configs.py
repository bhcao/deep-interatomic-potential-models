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

from pathlib import Path

import pydantic
from omegaconf import ListConfig
from pydantic import field_validator, model_validator
from typing_extensions import Annotated, Self

PositiveInt = Annotated[int, pydantic.Field(gt=0)]
PositiveFloat = Annotated[float, pydantic.Field(gt=0)]
Percentage = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


class ChemicalSystemsReaderConfig(pydantic.BaseModel):
    """Pydantic-based config related to data preprocessing and loading into
    `ChemicalSystem`s.

    When directories are given in `*_dataset_paths`, files ending with `.hdf5` or `.h5`
    in those directories will be automatically detected and added.

    In `dataset_splits` and `*_num_to_load`, if a float is given, it will be interpreted as
    percentage. If an integer is given, it will be interpreted as number of data points.

    Attributes:
        train_dataset_paths: Path(s) to where the training set(s) are located.
                            Cannot be empty.
                            Will be converted to a list after validation.
        valid_dataset_paths: Path(s) to where the validation set(s) are located.
                            This can be empty.
                            Will be converted to a list after validation.
        test_dataset_paths: Path(s) to where the test set(s) are located.
                            This can be empty.
                            Will be converted to a list after validation.
        dataset_splits: Split train dataset(s) into train, validation and test datasets.
                        Cannot be provided if ``valid_dataset_paths`` or ``test_dataset_paths``
                        are not empty.
                        If ``None``, then no splitting will be done.
        shuffle: Whether to shuffle the data before splitting and loading. Default is ``True``.
        parallel: Whether to use parallel loading or not. Every dataset file will use a
                  separate process to load data. Default is ``True``.
        train_num_to_load: Number of training set data points to load from the given
                           dataset. By default, this is ``None`` which means all the
                           data points are loaded.
                           If multiple dataset paths are given, this limit will apply in total.
        valid_num_to_load: Number of validation set data points to load from the given
                           dataset. By default, this is ``None`` which means all the
                           data points are loaded.
                           If multiple dataset paths are given, this limit will apply in total.
        test_num_to_load: Number of test set data points to load from the given
                           dataset. By default, this is ``None`` which means all the
                           data points are loaded.
                           If multiple dataset paths are given, this limit will apply in total.
    """

    train_dataset_paths: str | Path | list[str | Path]
    valid_dataset_paths: str | Path | list[str | Path] | None = None
    test_dataset_paths: str | Path | list[str | Path] | None = None

    dataset_splits: (
        tuple[PositiveInt, PositiveInt, PositiveInt] |
        tuple[Percentage, Percentage, Percentage] | None
    ) = None

    shuffle: bool = True
    parallel: bool = True

    train_num_to_load: PositiveInt | Percentage | None = None
    valid_num_to_load: PositiveInt | Percentage | None = None
    test_num_to_load: PositiveInt | Percentage | None = None

    @field_validator(
        "train_dataset_paths",
        "valid_dataset_paths",
        "test_dataset_paths",
        mode="before",
    )
    @classmethod
    def expand_path_to_list(
        cls, value: str | Path | list[str | Path] | ListConfig | None
    ) -> list[str | Path]:
        """Converts a single path to a list of paths and expands directories."""
        if value is None:
            return []

        if isinstance(value, (str, Path)):
            paths = [value]
        elif isinstance(value, list):
            paths = value
        elif isinstance(value, ListConfig):
            paths = list(value)
        else:
            raise ValueError(
                f"*_dataset_paths must be a string, Path, or a list of them, "
                f"but was {type(value)} - {value}"
            )

        expanded_paths = []
        for path in paths:
            if isinstance(path, str):
                path = Path(path)
            if path.is_dir():
                for file in path.glob("*.hdf5"):
                    expanded_paths.append(file)
                for file in path.glob("*.h5"):
                    expanded_paths.append(file)
            else:
                expanded_paths.append(path)
        return expanded_paths


    @model_validator(mode="after")
    def validate_dataset_paths(self) -> Self:
        """Validates the dataset paths and splits."""

        if self.train_dataset_paths == []:
            raise ValueError("Train dataset paths should contain at least one path")

        if self.dataset_splits is not None:
            if self.valid_dataset_paths is not None:
                raise ValueError(
                    "Cannot provide both `dataset_splits` and `valid_dataset_paths`. "
                    "Please provide only one of them."
                )
            if self.test_dataset_paths is not None:
                raise ValueError(
                    "Cannot provide both `dataset_splits` and `test_dataset_paths`. "
                    "Please provide only one of them."
                )
        return self


class GraphDatasetBuilderConfig(pydantic.BaseModel):
    """Pydantic-based config related to graph dataset building and preprocessing.

    Attributes:
        graph_cutoff_angstrom: Graph cutoff distance in Angstrom to apply when
                               creating the graphs. Default is 5.0.
        max_n_node: This value will be multiplied with the batch size to determine the
                    maximum number of nodes we allow in a batch.
                    Note that a batch will always contain max_n_node * batch_size
                    nodes, as the remaining ones are filled up with dummy nodes.
                    If set to `None`, a reasonable value will be automatically
                    computed. Default is `None`.
        max_n_edge: This value will be multiplied with the batch size to determine the
                    maximum number of edges we allow in a batch.
                    Note that a batch will always contain max_n_edge * batch_size
                    edges, as the remaining ones are filled up with dummy edges.
                    If set to `None`, a reasonable value will be automatically
                    computed. Default is `None`.
        batch_size: The number of graphs in a batch. Will be filled up with dummy graphs
                    if either the maximum number of nodes or edges are reached before
                    the number of graphs is reached. Default is 16.
        num_batch_prefetch: Number of batched graphs to prefetch while iterating
                            over batches. Default is 1.
        batch_prefetch_num_devices: Number of threads to use for prefetching.
                                    Default is 1.
        use_formation_energies: Whether the energies in the dataset should already be
                                transformed to subtract the average atomic energies.
                                Default is ``False``. Make sure that if you set this
                                to ``True``, the models assume ``"zero"`` atomic
                                energies as can be set in the model hyperparameters.
        atomic_energies_map: The atomic energies map to use for the formation energies.
                             If not provided, the atomic energies will be computed
                             from the dataset.
        avg_num_neighbors: The pre-computed average number of neighbors.
        avg_num_nodes: The pre-computed average number of nodes per graph.
        avg_r_min_angstrom: The pre-computed average minimum distance between nodes.

    """

    graph_cutoff_angstrom: PositiveFloat = 5.0
    max_n_node: PositiveInt | None = None
    max_n_edge: PositiveInt | None = None
    batch_size: PositiveInt = 16

    num_batch_prefetch: PositiveInt = 1
    batch_prefetch_num_devices: PositiveInt = 1

    use_formation_energies: bool = False
    atomic_energies_map: dict[int, float] | None = None
    avg_num_neighbors: float | None = None
    avg_num_nodes: float | None = None
    avg_r_min_angstrom: float | None = None
