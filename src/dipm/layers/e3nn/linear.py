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

'''
Modified from e3nn_jax._src.linear_flax to support nnx
'''

from collections.abc import Callable

from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Dtype, Initializer
import jax.numpy as jnp

import e3nn_jax as e3nn

from e3nn_jax._src.linear import (
    FunctionalLinear,
    validate_inputs_for_instructions,
)


class Linear(nnx.Module):
    r"""Equivariant Linear Flax module, modified from `e3nn_jax._src.linear_flax.Linear`.

    Args:
        irreps_out (`Irreps`): output representations, if allowed bu Schur's lemma.
        channel_out (optional int): if specified, the last axis before the irreps
            is assumed to be the channel axis and is mixed with the irreps.
        irreps_in (optional `Irreps`): input representations. If not specified,
            the input representations is obtained when calling the module.
        biases (bool): whether to add a bias to the output.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
        weights_per_channel (bool): whether to have one set of weights per channel.
        force_irreps_out (bool): whether to force the output irreps to be the one specified in ``irreps_out``.
    
    NOTE: We only support vanilla here, external weights and indexed weights are not supported.
    """

    def __init__(
        self,
        irreps_in: e3nn.Irreps | str,
        irreps_out: e3nn.Irreps | str,
        channel_out: int | None = None,
        gradient_normalization: float | str | None = None,
        path_normalization: float | str | None = None,
        biases: bool = False,
        parameter_initializer: Callable[[], Initializer] | None = None,
        instructions: list[tuple[int, int]] | None = None,
        force_irreps_out: bool = False,
        simplify_irreps_internally: bool = True,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        irreps_in = e3nn.Irreps(irreps_in)
        irreps_out = e3nn.Irreps(irreps_out)

        if simplify_irreps_internally:
            irreps_out = irreps_out.simplify()

        if not force_irreps_out:
            irreps_out = irreps_out.filter(keep=irreps_in)

        if channel_out is not None:
            irreps_out = channel_out * irreps_out

        self.linear_fn = FunctionalLinear(
            irreps_in,
            irreps_out,
            biases=biases,
            instructions=instructions,
            path_normalization=path_normalization,
            gradient_normalization=gradient_normalization,
        )

        def get_parameter(
            path_shape: tuple[int, ...],
            weight_std: float,
        ):
            # Default is to initialize the weights with a normal distribution.
            if parameter_initializer is None:
                parameter_initializer_ = initializers.normal(stddev=weight_std)
            else:
                parameter_initializer_ = parameter_initializer()

            key = rngs.params()
            return nnx.Param(parameter_initializer_(key, path_shape, param_dtype))

        self.kernels = [
            get_parameter(
                ins.path_shape,
                ins.weight_std,
            )
            for ins in self.linear_fn.instructions
        ]

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.channel_out = channel_out
        self.instructions = instructions
        self.force_irreps_out = force_irreps_out
        self.simplify_irreps_internally = simplify_irreps_internally


    def __call__(self, inputs: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        """Apply the linear operator.

        Args:
            inputs (IrrepsArray): input irreps-array of shape ``(..., [channel_in,] irreps_in.dim)``.
                Broadcasting with `weights` is supported.

        Returns:
            IrrepsArray: output irreps-array of shape ``(..., [channel_out,] irreps_out.dim)``.
                Properly normalized assuming that the weights and input are properly normalized.
        """
        if self.simplify_irreps_internally:
            inputs = inputs.remove_zero_chunks().regroup()

        if self.channel_out is not None:
            inputs = inputs.axis_to_mul()

        validate_inputs_for_instructions(
            inputs,
            self.instructions,
            self.simplify_irreps_internally,
            self.channel_out,
            self.irreps_in,
        )

        kernels = [param.value for param in self.kernels]

        def f(x):
            return self.linear_fn(kernels, x)
        for _ in range(inputs.ndim - 1):
            f = e3nn.utils.vmap(f)

        output = f(inputs)

        if self.channel_out is not None:
            output = output.mul_to_axis(self.channel_out)

        if self.force_irreps_out:
            return output.rechunk(self.irreps_out)
        return output
