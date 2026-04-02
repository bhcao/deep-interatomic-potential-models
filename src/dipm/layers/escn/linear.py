# Copyright 2025 Zhongguancun Academy
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

from dipm.layers.escn.utils import get_expand_index


class SO3LinearV2(nnx.Module):
    """EquiformerV2 linear layer."""
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

        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.dtype = dtype

    def __call__(self, embedding: jax.Array):
        kernel, bias, embedding = dtypes.promote_dtype(
            (self.kernel.value, self.bias.value, embedding), dtype=self.dtype
        )

        expand_index = get_expand_index(self.lmax)

        weight_expanded = kernel[expand_index] # [(L_max + 1) ** 2, C_in, C_out]
        out = jnp.einsum(
            "bmi, mio -> bmo", embedding, weight_expanded
        )  # [N, (L_max + 1) ** 2, C_out]
        out = out.at[:, 0:1, :].add(
            bias.reshape(1, 1, self.out_features)
        )

        return out


class MoLE(nnx.Module):
    """Mixture-of-Experts linear layer used in UMA."""
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
        self.cached_kernel = nnx.data(None)
        self.decode = False

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_bias = use_bias
        self.dtype = dtype

    def _get_kernel(self, expert_coeffs: jax.Array) -> jax.Array:
        kernel_moe, expert_coeffs = dtypes.promote_dtype(
            (self.kernel.value, expert_coeffs), dtype=self.dtype
        )

        kernel = jnp.einsum(
            "eio,be->bio",
            kernel_moe,
            expert_coeffs,
        )
        return kernel

    def __call__(
        self,
        inputs: jax.Array,
        n_node: jax.Array,
        *,
        expert_coeffs: jax.Array | None = None,
    ):
        if self.decode:
            assert self.cached_kernel is not None, (
                "nnx.view must be called before performing inference"
            )
            kernel = self.cached_kernel.value
        else:
            assert expert_coeffs is not None, (
                "expert_coeffs must be provided in training mode"
            )
            kernel = self._get_kernel(expert_coeffs)

        bias = self.bias.value if self.bias is not None else None
        inputs, bias = dtypes.promote_dtype(
            (inputs, bias), dtype=self.dtype
        )

        if len(n_node) == 1:
            result = jnp.einsum("b...i,io->b...o", inputs, kernel[0])
        else:
            # TODO(bhcao): Very slow, but in jax.jit dynamic slice is not supported. Consider to
            # align shape of every sample in th batch then apply reshape.
            kernel = jnp.repeat(kernel, n_node, total_repeat_length=len(inputs), axis=0)
            result = jnp.einsum("b...i,bio->b...o", inputs, kernel)

        if bias is not None:
            result += bias
        return result

    def set_view(
        self,
        decode: bool | None = None,
        expert_coeffs: jax.Array | None = None,
        **kwargs,
    ) -> dict:
        """Class method used by ``nnx.view``.

        Args:
            decode: If True, the module is set to decode mode.
            expert_coeffs: The expert mixing coefficients to get kernel weights.
        """

        if decode is not None:
            self.decode = decode

            if decode:
                assert expert_coeffs is not None, (
                    "expert_coeffs must be provided in decode mode"
                )
                kernel = self._get_kernel(expert_coeffs)
                self.cached_kernel = nnx.Cache(kernel)

        return kwargs
