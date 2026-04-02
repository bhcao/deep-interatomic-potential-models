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

import re
from typing import Any

import jax
from flax import nnx
from flax.typing import Dtype
import pydantic

from dipm.data.dataset_info import DatasetInfo
from dipm.typing import get_dtype, DtypeEnum


class ForceModelConfig(pydantic.BaseModel):
    """Base class for GNN node-wise energy models configuration.

    This class defines common hyperparameters. It must be overridden by
    the child classes.

    Attributes:
        force_head: Whether to predict forces with forces head. Default is ``False``.
        param_dtype: The data type of model parameters. Default is ``jnp.float32``.
    """

    force_head: bool = False
    param_dtype: DtypeEnum = DtypeEnum.F32


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

    The ``jax.Array`` constant should be defined in ``__call__`` instead of in
    ``__init__``, or ``jax.jit`` will treat it as a runtime buffer rather than a
    compile time literal. In case of duplicate creation when JIT is disabled, you
    can use ``functools.cache`` to cache it.

    To support direct forces prediction, you should specify ``force_head_prefix``
    (str) in the class's constants. The prefix should be the state dict key
    prefix for the forces head parameters.

    To support dropping unseen elements while loading the model, you should
    specify the ``embedding_layer_regexp`` (re.Pattern) attribute in the class's
    constants. The pattern will be used to match the embedding layer parameters
    of shape (num_species, ...) and drop the corresponding rows of the
    that should be modified.

    The number of elements (atomic species descriptors) allowed will always be
    inferred from the atomic energies map in the dataset info.
    """

    Config = ForceModelConfig  # Must be overridden by the child classes
    force_head_prefix: str | None = None
    embedding_layer_regexp: re.Pattern | None = None

    def __init__(
        self,
        config: dict | Config,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
    ):
        self.config = self.Config(**config) if isinstance(config, dict) else config
        self.dataset_info = dataset_info
        self.param_dtype = get_dtype(self.config.param_dtype)
        self.dtype = dtype or self.param_dtype

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        n_node: jax.Array,
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
