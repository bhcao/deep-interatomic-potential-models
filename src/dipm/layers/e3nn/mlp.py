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
from flax.nnx.nn import initializers
from flax.typing import Dtype
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from dipm.layers.activations import get_activation_fn


class MultiLayerPerceptron(nnx.Module):
    r"""Just a simple MLP for scalars. No equivariance here. Last layer will have no activation and no normalization.

    Args:
        list_neurons (list of int): number of neurons in each layer (excluding the input layer)
        act (optional callable): activation function
        gradient_normalization (str or float): normalization of the gradient

            - "element": normalization done in initialization variance of the weights, (the default in pytorch)
                gives the same importance to each neuron, a layer with more neurons will have a higher importance
                than a layer with less neurons
            - "path" (default): normalization done explicitly in the forward pass,
                gives the same importance to every layer independently of the number of neurons
        
        For compatibility with NequIP MLP, set gradient_normalization=False, use_act_norm=False and specify scalar_mlp_std.
        If gradient_normalization=True, and scalar_mlp_std=1.0 (by default), which corresponds to the e3nn version.
    """

    def __init__(
        self,
        features_list: list[int],
        activation: str | None = None,
        use_layer_norm: bool = False,
        gradient_normalization: str | float | None = None,
        use_bias: bool = False,
        use_act_norm: bool = True,
        scalar_mlp_std: float = 1.0,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # Gradient normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]
        self.gradient_normalization = gradient_normalization

        act_norm = e3nn.normalize_function if use_act_norm else (lambda x: x)

        # Activation and normalization
        self.act = (
            (lambda x: x)
            if activation is None
            else act_norm(get_activation_fn(activation))
        )

        # Layers
        self.layers = []
        self.norms = []
        in_features = features_list[0]
        for i, out_features in enumerate(features_list[1:]):
            scale = scalar_mlp_std if i < len(features_list) - 2 else 1.0
            stddev = (scale / jnp.sqrt(in_features)) ** (1.0 - gradient_normalization)
            layer = nnx.Linear(
                in_features,
                out_features,
                use_bias=use_bias,
                kernel_init=initializers.normal(stddev),
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.layers.append(layer)
            if use_layer_norm and i < len(features_list) - 2:
                self.norms.append(nnx.LayerNorm(out_features, param_dtype=param_dtype, rngs=rngs))
            else:
                self.norms.append(lambda x: x) # placeholder
            in_features = out_features


    def __call__(self, x: jax.Array | e3nn.IrrepsArray) -> jax.Array | e3nn.IrrepsArray:
        """Evaluate the MLP

        Input and output are either `jax.Array` or `IrrepsArray`.
        If the input is a `IrrepsArray`, it must contain only scalars.

        Args:
            x (IrrepsArray): input of shape ``[..., input_size]``

        Returns:
            IrrepsArray: output of shape ``[..., list_neurons[-1]]``
        """
        if isinstance(x, e3nn.IrrepsArray):
            if not x.irreps.is_scalar():
                raise ValueError("MLP only works on scalar (0e) input.")
            x = x.array
            output_irrepsarray = True
        else:
            output_irrepsarray = False

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            alpha = 1 / x.shape[-1]
            x = jnp.sqrt(alpha) ** self.gradient_normalization * layer(x)
            if i < len(self.layers) - 1:
                x = self.act(norm(x))

        if output_irrepsarray:
            x = e3nn.IrrepsArray(e3nn.Irreps(f"{x.shape[-1]}x0e"), x)
        return x
