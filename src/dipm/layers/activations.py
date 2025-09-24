# Copyright 2025 InstaDeep Ltd and Cao Bohan
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

from enum import Enum
from collections.abc import Callable

import jax
from jax import Array
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer
from flax.nnx.nn import initializers


class BetaSwish(nnx.Module):
    """BetaSwish activation function, a learnable variant of Swish."""

    def __init__(
        self,
        features: int,
        *,
        param_dtype: Dtype = jnp.float32,
        beta_init: Initializer = initializers.zeros,
        rngs: nnx.Rngs,
    ):
        key = rngs.params()
        self.beta = nnx.Param(beta_init(key, (features,), param_dtype))
        self.features = features

    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta.value * x)


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
    param_dtype: Dtype = jnp.float32,
    rngs: nnx.Rngs | None = None,
) -> Callable[[Array], Array]:
    """Parse activation function among available options.

    See :class:`~dipm.layers.activations.Activation`.

    Args:
        act: Activation type.
        features (optional): Number of features only for BetaSwish.
        param_dtype (optional): Dtype of BetaSwish parameters.
        rngs (optional): Only for BetaSwish.
    """
    if Activation(act) == Activation.BETA_SWISH:
        assert features != -1, "Please specify the `features` parameter for BetaSwish."
        assert rngs is not None, "Please specify a nnx.Rngs for BetaSwish."
        return BetaSwish(features, param_dtype=param_dtype, rngs=rngs)

    activations_map = {
        Activation.TANH: jax.nn.tanh,
        Activation.SILU: jax.nn.silu,
        Activation.RELU: jax.nn.relu,
        Activation.ELU: jax.nn.elu,
        Activation.SIGMOID: jax.nn.sigmoid,
        Activation.NONE: lambda x: x,
    }
    return activations_map[Activation(act)]
