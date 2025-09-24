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

from typing import Any

import jax
from flax import nnx
import pydantic

from dipm.data.dataset_info import DatasetInfo


class ForceModel(nnx.Module):
    """Base class for GNN node-wise energy models.

    Energy models deriving from this class return node-wise
    contributions to the total energy, from the edge vectors of a graph,
    the atomic species of the nodes, and the edges themselves passed
    as `senders` and `receivers` indices.

    Our MLIP models are validated with Pydantic, and hold a reference to
    their `.Config` class describing the set of hyperparameters.

    All subclasses of ForceModel must call super.__init__() and pass in
    an additional nnx.Rngs parameter during initialization
    """

    Config = pydantic.BaseModel  # Must be overridden by the child classes

    def __init__(self, config: dict | Config, dataset_info: DatasetInfo):
        self.config = self.Config(**config) if isinstance(config, dict) else config
        self.dataset_info = dataset_info

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        """Compute node-wise energy summands. This function must be overridden by the
        implementation of `ForceModel`.
        """
        raise NotImplementedError(
            "No energy model defined by ForceModel.__call__, "
            "but must be overridden by its child classes."
        )

    def __init_subclass__(cls, **kwargs: Any):
        """This enforces that child classes will
        need to override the `Config` attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls.Config is pydantic.BaseModel:
            raise NotImplementedError(
                f"{cls.__name__} must override the `Config` attribute."
            )
