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
from pydantic import field_validator, model_validator
from typing_extensions import Self

from dipm.typing import PositiveInt, PositiveFloat, Proportion


def _expand_path_to_list(paths) -> list[Path]:
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    expanded = []
    for path in paths:
        path = Path(path).absolute()
        if path.is_dir():
            for file in path.glob("*.hdf5"):
                expanded.append(file)
            for file in path.glob("*.h5"):
                expanded.append(file)
        else:
            expanded.append(path)
    return expanded


class ChemicalDatasetsConfig(pydantic.BaseModel):
    """Pydantic-based config related to data preprocessing and loading into
    `ChemicalSystem`s.

    When directories are given in `*_dataset_paths`, files ending with `.hdf5` or `.h5`
    in those directories will be automatically detected and added. If dict is given,
    the keys will be used as task/dataset names and the values as paths.

    In `dataset_splits` and `*_num_to_load`, if a float is given, it will be interpreted as
    proportion. If an integer is given, it will be interpreted as number of data points.

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

    train_dataset_paths: list[Path] | dict[str, list[Path]]
    valid_dataset_paths: list[Path] | dict[str, list[Path]] | None = None
    test_dataset_paths: list[Path] | dict[str, list[Path]] | None = None

    dataset_splits: (
        tuple[PositiveInt, PositiveInt, PositiveInt] |
        tuple[Proportion, Proportion, Proportion] | None
    ) = None

    shuffle: bool = True
    parallel: bool = True

    train_num_to_load: PositiveInt | Proportion | None = None
    valid_num_to_load: PositiveInt | Proportion | None = None
    test_num_to_load: PositiveInt | Proportion | None = None

    @field_validator(
        "train_dataset_paths",
        "valid_dataset_paths",
        "test_dataset_paths",
        mode="before",
    )
    @classmethod
    def expand_path_to_dict(cls, value) -> list[Path] | dict[str, list[Path]] | None:
        """Converts a single path to a list of paths and expands directories."""
        if value is None:
            return None

        if not isinstance(value, dict):
            return _expand_path_to_list(value)

        expanded_paths = {}
        for key, paths in value.items():
            expanded_paths[key] = _expand_path_to_list(paths)
        return expanded_paths


    @model_validator(mode="after")
    def validate_dataset_paths(self) -> Self:
        """Validates the dataset paths and splits."""

        if isinstance(self.train_dataset_paths, dict):
            dataset_names = set(self.train_dataset_paths.keys())
            if isinstance(self.valid_dataset_paths, list) or (
                self.valid_dataset_paths is not None and
                set(self.valid_dataset_paths.keys()) != dataset_names
            ):
                raise ValueError(
                    "Your `train_dataset_paths` is a dictionary, but `valid_dataset_paths` "
                    "is a list or has different keys."
                )
            if isinstance(self.test_dataset_paths, list) or (
                self.test_dataset_paths is not None and
                set(self.test_dataset_paths.keys()) != dataset_names
            ):
                raise ValueError(
                    "Your `train_dataset_paths` is a dictionary, but `test_dataset_paths` "
                    "is a list or has different keys."
                )
        else:
            if isinstance(self.valid_dataset_paths, dict):
                raise ValueError(
                    "Your `train_dataset_paths` is a list, but `valid_dataset_paths` "
                    "is a dictionary."
                )
            if isinstance(self.test_dataset_paths, dict):
                raise ValueError(
                    "Your `train_dataset_paths` is a list, but `test_dataset_paths` "
                    "is a dictionary."
                )

        if self.train_dataset_paths in [[], {}]:
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
        max_neighbors_per_atom: The maximum number of neighbors to consider for each atom.
                                If None, all neighbors within the cutoff will be considered.
                                Default is `None`.
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
        drop_unseen_elements: If ``dataset_info`` is provided, whether to drop unseen
                              elements in the training dataset from the ``dataset_info``
                              atomic numbers table. If ``True``, remember to remove unused
                              embeddings from the model by yourself. Default is ``False``.
    """

    graph_cutoff_angstrom: PositiveFloat = 5.0
    max_n_node: PositiveInt | None = None
    max_n_edge: PositiveInt | None = None
    max_neighbors_per_atom: PositiveInt | None = None
    batch_size: PositiveInt = 16

    num_batch_prefetch: PositiveInt = 1
    batch_prefetch_num_devices: PositiveInt = 1

    use_formation_energies: bool = False
    drop_unseen_elements: bool = False
