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

import jax.numpy as jnp
from flax import nnx

from dipm.layers.escn.transform import get_s2grid_mats
from dipm.layers.escn.utils import get_expand_index


class GateActivation(nnx.Module):
    """Apply gate for vector and silu for scalar."""

    def __init__(self, lmax: int, mmax: int, num_channels: int, m_prime: bool = False):
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels
        self.m_prime = m_prime

    def __call__(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        # Can be used directly on m_prime representation
        expand_index = get_expand_index(
            self.lmax, self.mmax, vector_only=True, m_prime=self.m_prime
        )

        gating_scalars = nnx.sigmoid(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )[:, expand_index]

        input_tensors_scalars = nnx.silu(input_tensors[:, 0:1])
        input_tensors_vectors = input_tensors[:, 1:] * gating_scalars
        output_tensors = jnp.concat(
            (input_tensors_scalars, input_tensors_vectors), axis=1
        )

        return output_tensors


class S2Activation(nnx.Module):
    """Apply silu on sphere function."""

    def __init__(self, lmax: int, mmax: int, resolution: int, m_prime: bool = False):
        self.lmax = lmax
        self.mmax = mmax
        self.resolution = resolution
        self.m_prime = m_prime

    def __call__(self, inputs):
        so3_grid = get_s2grid_mats(
            self.lmax, self.mmax, resolution=self.resolution, m_prime=self.m_prime
        )

        x_grid = so3_grid.to_grid(inputs)
        x_grid = nnx.silu(x_grid)
        outputs = so3_grid.from_grid(x_grid)
        return outputs


class SeparableS2Activation(nnx.Module):
    """Apply silu on sphere function for vector and silu directly for scalar."""

    def __init__(self, lmax: int, mmax: int, resolution: int, m_prime: bool = False):
        self.s2_act = S2Activation(lmax, mmax, resolution, m_prime)

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
