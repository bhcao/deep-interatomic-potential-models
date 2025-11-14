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

from enum import Enum
from collections.abc import Callable

import jax
from jax import Array
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer
from flax.nnx.nn import initializers, dtypes


class SmoothLeakyReLU(nnx.Module):
    '''Smooth Leaky ReLU activation.'''
    def __init__(self, negative_slope: float = 0.2):
        self.alpha = negative_slope

    def __call__(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * nnx.sigmoid(x) - 1)
        return x1 + x2


class BetaSwish(nnx.Module):
    """BetaSwish activation function, a learnable variant of Swish."""

    def __init__(
        self,
        features: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        beta_init: Initializer = initializers.zeros,
        rngs: nnx.Rngs,
    ):
        key = rngs.params()
        self.beta = nnx.Param(beta_init(key, (features,), param_dtype))
        self.features = features
        self.dtype = dtype

    def __call__(self, x):
        x, beta = dtypes.promote_dtype((x, self.beta.value), dtype=self.dtype)
        return x * jax.nn.sigmoid(beta * x)


# --- Activation options ---


class Activation(Enum):
    """Supported activation functions:

    Options are:
    `TANH = "tanh"`,
    `SILU = "silu"`,
    `RELU = "relu"`,
    `ELU = "elu"`,
    `BETA_SWISH = "beta_swish"`,
    `SIGMOID = "sigmoid"`, and
    `NONE = "none"`.
    """

    TANH = "tanh"
    SILU = "silu" # swish is alias of silu
    RELU = "relu"
    ELU = "elu"
    BETA_SWISH = "beta_swish" # learnable
    SIGMOID = "sigmoid"
    NONE = "none"


def get_activation_fn(
    act: Activation | str,
    features: int = -1,
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    rngs: nnx.Rngs | None = None,
) -> Callable[[Array], Array]:
    """Parse activation function among available options.

    See :class:`~dipm.layers.activations.Activation`.

    Args:
        act: Activation type.
        features (optional): Number of features only for BetaSwish.
        dtype (optional): Data type of BetaSwish during computation.
        param_dtype (optional): Dtype of BetaSwish parameters.
        rngs (optional): Only for BetaSwish.
    """
    if Activation(act) == Activation.BETA_SWISH:
        assert features != -1, "Please specify the `features` parameter for BetaSwish."
        assert rngs is not None, "Please specify a nnx.Rngs for BetaSwish."
        return BetaSwish(features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    activations_map = {
        Activation.TANH: jax.nn.tanh,
        Activation.SILU: jax.nn.silu,
        Activation.RELU: jax.nn.relu,
        Activation.ELU: jax.nn.elu,
        Activation.SIGMOID: jax.nn.sigmoid,
        Activation.NONE: lambda x: x,
    }
    return activations_map[Activation(act)]
