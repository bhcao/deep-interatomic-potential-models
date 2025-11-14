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
