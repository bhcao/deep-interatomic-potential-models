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

from flax import nnx
from flax.nnx.module import first_from
import jax
import jax.numpy as jnp


def drop_path(inputs: jax.Array, drop_prob: float = 0.0, *, rngs: nnx.RngStream) -> jax.Array:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
    key = rngs()
    random_tensor = keep_prob + jax.random.uniform(key, shape, dtype=inputs.dtype)
    random_tensor = jnp.floor(random_tensor)  # binarize
    output = (inputs / keep_prob) * random_tensor
    return output


class GraphDropPath(nnx.Module):
    """Consider batch for graph inputs when dropping paths."""

    def __init__(
        self,
        drop_prob: float,
        *,
        deterministic: bool = False,
    ) -> None:
        self.drop_prob = drop_prob
        self.deterministic = deterministic

    def __call__(
        self,
        inputs: jax.Array,
        n_node: jax.Array,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to GraphDropPath
                as either a __call__ argument or class attribute""",
        )

        if (self.drop_prob == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.drop_prob == 1.0:
            return jnp.zeros_like(inputs)

        batch_size = len(n_node)
        # work with diff dim tensors, not just 2D ConvNets
        shape = (batch_size,) + (1,) * (inputs.ndim - 1)
        ones = jnp.ones(shape, dtype=inputs.dtype)
        drop = drop_path(ones, self.drop_prob, rngs=rngs['dropout'])

        # create pyg batch from n_node
        output_size = n_node.shape[0]
        num_elements = inputs.shape[0]
        batch = jnp.repeat(jnp.arange(output_size), n_node, total_repeat_length=num_elements)

        out = inputs * drop[batch]
        return out
