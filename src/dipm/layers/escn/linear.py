# Copyright 2025 Cao Bohan
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

from flax import nnx
from flax.typing import Dtype, Initializer
from flax.nnx.nn import initializers
import jax
import jax.numpy as jnp

from dipm.layers.escn.utils import expand_index
from dipm.models.force_model import PrecallInterface


class SO3LinearV2(nnx.Module):
    '''EquiFormerV2 linear layer.'''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax: int,
        *,
        kernel_init: Initializer = initializers.lecun_normal(),
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        key = rngs.params()
        self.weight = nnx.Param(
            kernel_init(
                key, ((lmax + 1), in_features, out_features), param_dtype
            )
        )
        key = rngs.params()
        self.bias = nnx.Param(initializers.zeros(key, out_features, param_dtype))

        self.expand_index = expand_index(lmax)

        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

    def __call__(self, embedding: jax.Array):
        weight = self.weight.value[self.expand_index] # [(L_max + 1) ** 2, C_in, C_out]
        out = jnp.einsum(
            "bmi, mio -> bmo", embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        out = out.at[:, 0:1, :].add(
            self.bias.value.reshape(1, 1, self.out_features)
        )

        return out


class MoLE(nnx.Module, PrecallInterface):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.kernel = nnx.Param(
            initializers.lecun_uniform()(
                rngs.params(), (num_experts, in_features, out_features), param_dtype
            )
        )
        self.bias = nnx.Param(
            initializers.zeros(rngs.params(), (out_features,), param_dtype)
        ) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_bias = use_bias

    # pylint: disable=arguments-differ
    def cache(self, expert_mixing_coeffs: jax.Array, n_node: jax.Array, **_kwargs):
        kernel = jnp.einsum(
            "eio,be->bio",
            self.kernel.value,
            expert_mixing_coeffs,
        )

        return {"kernel": kernel, "n_node": n_node}

    @PrecallInterface.context_handler
    def __call__(self, inputs: jax.Array, *, kernel: jax.Array, n_node: jax.Array):
        """Kernel and n_node will be automatically added by context handler from cache."""
        bias = self.bias.value if self.bias is not None else None

        # TODO(bhcao): Very slow, but in jax.jit dinamic slice is not supported. Consider to align
        # shape of every sample in th batch then apply reshape.
        batch = jnp.repeat(jnp.arange(len(n_node)), n_node, total_repeat_length=len(inputs))
        result = jnp.einsum("b...i,bio->b...o", inputs, kernel[batch])

        if bias is not None:
            result += bias
        return result
