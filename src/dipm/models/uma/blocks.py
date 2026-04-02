# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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

from enum import Enum

from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers, dtypes
import jax
import jax.numpy as jnp

from dipm.layers.escn import (
    get_s2grid_mats,
    GateActivation,
    SeparableS2Activation,
    WignerMats,
    SO3LinearV2,
    SO2Convolution,
)


class ActivationType(Enum):
    """Type of attention activation."""
    GATE = "gate"
    S2_SEP = "s2_sep"


class FeedForwardType(Enum):
    """Type of feed-forward layer."""
    SPECTRAL = "spectral"
    GRID = "grid"


class Edgewise(nnx.Module):
    def __init__(
        self,
        lmax: int,
        mmax: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels_list: list[int],
        grid_resolution: int,
        act_type: ActivationType = ActivationType.GATE,
        num_experts: int = 0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if act_type == ActivationType.GATE:
            self.act = GateActivation(
                lmax, mmax, hidden_channels, m_prime=True
            )
            extra_m0_output_channels = lmax * hidden_channels
        else:
            # This is the only place where the SO3 grid of the edges (lmax/mmax) is used
            self.act = SeparableS2Activation(lmax, mmax, grid_resolution, m_prime=True)
            extra_m0_output_channels = hidden_channels

        self.so2_conv_1 = SO2Convolution(
            lmax,
            mmax,
            2 * sphere_channels,
            hidden_channels,
            internal_weights=False,
            edge_channels_list=edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
            num_experts=num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.so2_conv_2 = SO2Convolution(
            lmax,
            mmax,
            hidden_channels,
            sphere_channels,
            num_experts=num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jax.Array, # In m primary mode
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMats,
        edge_envelope: jax.Array,
        *,
        n_node: jax.Array | None = None,
        expert_coeffs: jax.Array | None = None,
    ):
        num_nodes = len(node_feats)

        messages = jnp.concat((node_feats[senders], node_feats[receivers]), axis=2)

        # Rotate the irreps to align with the edge
        messages = wigner_matrices.rotate(messages)

        # SO2 convolution
        messages, x_0_gating = self.so2_conv_1(
            messages, edge_embeds, n_node=n_node, expert_coeffs=expert_coeffs,
        )
        messages = self.act(x_0_gating, messages)
        messages = self.so2_conv_2(
            messages, edge_embeds, n_node=n_node, expert_coeffs=expert_coeffs,
        )
        messages = messages * edge_envelope

        # Rotate back the irreps
        messages = wigner_matrices.rotate_inv(messages)

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(messages, receivers, num_segments=num_nodes)
        return node_feats


class SpectralAtomwise(nnx.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.scalar_mlp = nnx.Sequential(
            nnx.Linear(
                sphere_channels,
                lmax * hidden_channels,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            nnx.silu,
        )

        self.so3_linear_1 = SO3LinearV2(
            sphere_channels, hidden_channels, lmax,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.act = GateActivation(lmax, lmax, hidden_channels)
        self.so3_linear_2 = SO3LinearV2(
            hidden_channels, sphere_channels, lmax,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(self, node_feats: jax.Array):
        gating_scalars = self.scalar_mlp(node_feats[:, 0:1])
        node_feats = self.so3_linear_1(node_feats)
        node_feats = self.act(gating_scalars, node_feats)
        node_feats = self.so3_linear_2(node_feats)
        return node_feats


class GridAtomwise(nnx.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        grid_resolution: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.lmax = lmax
        self.grid_resolution = grid_resolution

        self.grid_mlp = nnx.Sequential(
            nnx.Linear(sphere_channels, hidden_channels, use_bias=False,
                       dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_channels, hidden_channels, use_bias=False,
                       dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_channels, sphere_channels, use_bias=False,
                       dtype=dtype, param_dtype=param_dtype, rngs=rngs),
        )

    def __call__(self, node_feats):
        so3_grid = get_s2grid_mats(self.lmax, self.lmax, self.grid_resolution)

        node_feats_grid = so3_grid.to_grid(node_feats)
        node_feats_grid = self.grid_mlp(node_feats_grid)
        node_feats = so3_grid.from_grid(node_feats_grid)
        return node_feats


class ChargeSpinTaskEmbed(nnx.Module):
    """Embeds the charge, spin and task / dataset information."""
    def __init__(
        self,
        num_channels: int,
        num_tasks: int | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # [-100, 100]
        self.charge_emb = nnx.Embed(
            201, num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        # [0, 100]
        self.spin_emb = nnx.Embed(
            101, num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.num_tasks = num_tasks
        if num_tasks is not None:
            self.task_emb = nnx.Embed(
                num_tasks, num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

        # Embed + Linear is mathematically equivalent to one_hot + Linear or Embed + add + bias
        self.bias = nnx.Param(initializers.zeros(rngs.params(), (num_channels,), param_dtype))
        self.dtype = dtype

    def __call__(
        self,
        charge: jax.Array,
        spin: jax.Array,
        task: jax.Array | None = None,
    ):
        bias, = dtypes.promote_dtype((self.bias.value,), dtype=self.dtype)
        charge_emb = self.charge_emb(charge + 100)
        spin_emb = self.spin_emb(spin)
        embeddings = charge_emb + spin_emb

        if self.num_tasks is not None:
            task_emb = self.task_emb(task)
            embeddings += task_emb

        embeddings += bias
        return nnx.silu(embeddings)
