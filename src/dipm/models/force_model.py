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

from collections.abc import Sequence, Callable
from functools import wraps
from typing import Any

import jax
from flax import nnx
from flax.typing import Dtype
import pydantic

from dipm.data.dataset_info import DatasetInfo
from dipm.typing import get_dtype


class PrecallInterface:
    """Interface for pre-call functions."""
    precall_key: str # ctx key in precall context

    def init_precall_key(self):
        """Create precall keys for each submodule. Will be called by `ForceFieldPredictor`."""
        for key, value in self.__dict__.items():
            if isinstance(value, PrecallInterface):
                value.precall_key = key
                value.init_precall_key()
            elif isinstance(value, Sequence):
                for idx, subvalue in enumerate(value):
                    if isinstance(subvalue, PrecallInterface):
                        subvalue.precall_key = f"{key}.{idx}"
                        subvalue.init_precall_key()

    def precall(self, **kwargs) -> dict[str, Any]:
        """Pre-call function to be called before the forward pass. To reduce possible errors,
        please always use keyword arguments and use a `kwargs` to catch extra arguments.
        """
        ctx = {}
        cache_val = self.cache(**kwargs) # pylint: disable=assignment-from-none
        if cache_val is not None:
            ctx['.'] = cache_val
        for key, value in self.__dict__.items():
            if isinstance(value, PrecallInterface):
                ctx[key] = value.precall(**kwargs)
            elif isinstance(value, Sequence):
                ctx.update({
                    f'{key}.{idx}': subvalue.precall(**kwargs)
                    for idx, subvalue in enumerate(value) if isinstance(subvalue, PrecallInterface)
                })
        return ctx

    def cache(self, **_kwargs) -> dict[str, Any] | None:
        """Return a dict of values to be cached for the forward pass."""
        return None

    @staticmethod
    def context_handler(forward_fn: Callable) -> Callable:
        """Wraps __call__() to handle precall context."""

        @wraps(forward_fn)
        def wrapped(self, *args, ctx: dict[str, Any] | None = None, **kwargs):
            if ctx is None:
                return forward_fn(self, *args, **kwargs)

            if getattr(self, "precall_key", None) is not None:
                ctx = ctx[self.precall_key]

            cur_ctx = ctx.pop('.', None)
            if cur_ctx is not None:
                kwargs.update(cur_ctx)

            if len(ctx) > 0:
                return forward_fn(self, *args, ctx=ctx, **kwargs)
            return forward_fn(self, *args, **kwargs)
        return wrapped


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

    NOTE: Don't keep any jax.Array that need to be used in the forward pass
    directly in the class attributes, as they won't be replicated when
    parallelized. Instead, use the nnx.Cache to wrap them.

    To support direct forces prediction, you should add the ``force_head``
    (bool) attribute to your Config class and specify ``force_head_prefix``
    (str) in the class's constants. The prefix should be the state dict key
    prefix for the forces head parameters.
    """

    Config = pydantic.BaseModel  # Must be overridden by the child classes

    def __init__(
        self,
        config: dict | Config,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
    ):
        self.config = self.Config(**config) if isinstance(config, dict) else config
        self.dataset_info = dataset_info
        self.dtype = dtype or get_dtype(getattr(self.config, 'param_dtype', 'f32'))
        self.predict_forces = getattr(self.config, "force_head", False)

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
