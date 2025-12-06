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

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers, dtypes

from dipm.layers import get_activation_fn


class MultiHeadLinear(nnx.Module):
    """Standard nnx.Linear layer with selectable heads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        kernel_key = rngs.params()
        self.kernel = nnx.Param(
            initializers.lecun_normal()(
                kernel_key, (num_heads, in_features, out_features), param_dtype
            )
        )
        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            bias_key = rngs.params()
            self.bias = nnx.Param(initializers.zeros(
                bias_key, (num_heads, out_features), param_dtype
            ))
        else:
            self.bias = nnx.data(None)

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype

    def __call__(self, inputs: jax.Array, n_node: jax.Array, head: jax.Array) -> jax.Array:
        bias = self.bias.value if self.bias is not None else None

        inputs, kernel, bias = dtypes.promote_dtype(
            (inputs, self.kernel.value, bias), dtype=self.dtype
        )

        kernel = jnp.repeat(kernel[head], n_node, total_repeat_length=len(inputs), axis=0)
        y = jnp.einsum("b...i,bio->b...o", inputs, kernel)

        assert self.use_bias == (bias is not None)
        if bias is not None:
            bias = jnp.repeat(bias[head], n_node, total_repeat_length=len(inputs), axis=0)
            y += jnp.reshape(bias, (len(bias),) + (1,) * (y.ndim - 2) + (-1,))
        return y


class EnergyHead(nnx.Module):
    """Two layer MLP with selectable heads for energy prediction."""

    def __init__(
        self,
        num_channels: int,
        num_heads: int = 1,
        activation: str = "silu",
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert num_heads > 0, "num_heads must be positive"

        linear_class = (
            functools.partial(MultiHeadLinear, num_heads=num_heads)
            if num_heads > 1 else nnx.Linear
        )
        self.num_heads = num_heads

        self.lin_in = linear_class(
            num_channels,
            num_channels // 2,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.act = get_activation_fn(activation)
        self.lin_out = linear_class(
            num_channels // 2,
            1,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, n_node: jax.Array, task: jax.Array | None = None):
        if self.num_heads == 1:
            x = self.lin_in(inputs)
            x = self.act(x)
            x = self.lin_out(x)
            return x

        assert task is not None, "task must be specified for multi-head energy head"

        x = self.lin_in(inputs, n_node, task)
        x = self.act(x)
        x = self.lin_out(x, n_node, task)
        return x
