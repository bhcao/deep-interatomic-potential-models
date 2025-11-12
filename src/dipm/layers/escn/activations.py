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

import jax
import jax.numpy as jnp
from flax import nnx

from dipm.layers.escn.transform import SO3Grid
from dipm.layers.escn.utils import expand_index


class GateActivation(nnx.Module):
    '''Apply gate for vector and silu for scalar.'''

    def __init__(self, lmax: int, mmax: int, num_channels: int, m_prime: bool = False):
        self.lmax = lmax
        self.num_channels = num_channels

        # Can be used directly on m_prime representation
        self.expand_index = nnx.Cache(expand_index(lmax, mmax, vector_only=True, m_prime=m_prime))

    def __call__(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        gating_scalars = nnx.sigmoid(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )[:, self.expand_index.value]

        input_tensors_scalars = nnx.silu(input_tensors[:, 0:1])
        input_tensors_vectors = input_tensors[:, 1:] * gating_scalars
        output_tensors = jnp.concat(
            (input_tensors_scalars, input_tensors_vectors), axis=1
        )

        return output_tensors


class S2Activation(nnx.Module):
    """Apply silu on sphere function."""

    def __init__(self, so3_grid: SO3Grid, perm: jax.Array | None = None):
        # activation in l_prime representation
        if perm is None:
            self.so3_grid = so3_grid
        else:
            # activation in m_prime representation
            self.so3_grid = so3_grid.to_m_prime_format(perm)

    def __call__(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = nnx.silu(x_grid)
        outputs = self.so3_grid.from_grid(x_grid)
        return outputs


class SeparableS2Activation(nnx.Module):
    """Apply silu on sphere function for vector and silu directly for scalar."""

    def __init__(self, so3_grid: SO3Grid, perm: jax.Array | None = None):
        self.s2_act = S2Activation(so3_grid, perm)

    def __call__(self, input_scalars, input_tensors):
        output_scalars = nnx.silu(input_scalars)
        output_tensors = self.s2_act(input_tensors)
        outputs = jnp.concat(
            (
                output_scalars[:, None],
                output_tensors[:, 1 : output_tensors.shape[1]],
            ),
            axis=1,
        )
        return outputs
