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

from flax import nnx
from flax.typing import Dtype, Initializer
from flax.nnx.nn import initializers, dtypes
import jax
import jax.numpy as jnp

from dipm.layers.escn.utils import expand_index
from dipm.models.force_model import PrecallInterface


class SO3LinearV2(nnx.Module):
    '''EquiformerV2 linear layer.'''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax: int,
        *,
        kernel_init: Initializer = initializers.lecun_normal(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        key = rngs.params()
        self.kernel = nnx.Param(
            kernel_init(
                key, ((lmax + 1), in_features, out_features), param_dtype
            )
        )
        key = rngs.params()
        self.bias = nnx.Param(initializers.zeros(key, out_features, param_dtype))

        self.expand_index = nnx.Cache(expand_index(lmax))

        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.dtype = dtype

    def __call__(self, embedding: jax.Array):
        kernel, bias, embedding = dtypes.promote_dtype(
            (self.kernel.value, self.bias.value, embedding), dtype=self.dtype
        )

        weight_expanded = kernel[self.expand_index.value] # [(L_max + 1) ** 2, C_in, C_out]
        out = jnp.einsum(
            "bmi, mio -> bmo", embedding, weight_expanded
        )  # [N, (L_max + 1) ** 2, C_out]
        out = out.at[:, 0:1, :].add(
            bias.reshape(1, 1, self.out_features)
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
        dtype: Dtype | None = None,
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
        self.dtype = dtype

    # pylint: disable=arguments-differ
    def cache(self, expert_mixing_coeffs: jax.Array, n_node: jax.Array, **_kwargs):
        kernel_moe, expert_mixing_coeffs = dtypes.promote_dtype(
            (self.kernel.value, expert_mixing_coeffs), dtype=self.dtype
        )

        kernel = jnp.einsum(
            "eio,be->bio",
            kernel_moe,
            expert_mixing_coeffs,
        )

        return {"kernel": kernel, "n_node": n_node}

    @PrecallInterface.context_handler
    def __call__(self, inputs: jax.Array, *, kernel: jax.Array, n_node: jax.Array):
        """Kernel and n_node will be automatically added by context handler from cache."""
        bias = self.bias.value if self.bias is not None else None
        inputs, kernel, bias = dtypes.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )

        # TODO(bhcao): Very slow, but in jax.jit dinamic slice is not supported. Consider to align
        # shape of every sample in th batch then apply reshape.
        batch = jnp.repeat(jnp.arange(len(n_node)), n_node, total_repeat_length=len(inputs))
        result = jnp.einsum("b...i,bio->b...o", inputs, kernel[batch])

        if bias is not None:
            result += bias
        return result
