# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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

from enum import Enum

from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers, dtypes
import jax
import jax.numpy as jnp

from dipm.layers.escn import (
    GateActivation,
    SeparableS2Activation,
    MappingCoefficients,
    SO3Grid,
    WignerMatrices,
    SO3LinearV2,
    SO2Convolution,
)
from dipm.models.force_model import PrecallInterface


class ActivationType(Enum):
    '''Type of attention activation.'''
    GATE = "gate"
    S2_SEP = "s2_sep"


class FeedForwardType(Enum):
    '''Type of feed-forward layer.'''
    SPECTRAL = "spectral"
    GRID = "grid"


class Edgewise(nnx.Module, PrecallInterface):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels_list: list[int],
        mapping_coeffs: MappingCoefficients,
        so3_grid: SO3Grid,
        act_type: ActivationType = ActivationType.GATE,
        num_experts: int = 0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if act_type == ActivationType.GATE:
            self.act = GateActivation(
                mapping_coeffs.lmax, mapping_coeffs.mmax, hidden_channels, m_prime=True
            )
            extra_m0_output_channels = mapping_coeffs.lmax * hidden_channels
        else:
            # This is the only place where the SO3 grid of the edges (lmax/mmax) is used
            self.act = SeparableS2Activation(so3_grid, mapping_coeffs.perm)
            extra_m0_output_channels = hidden_channels

        self.so2_conv_1 = SO2Convolution(
            2 * sphere_channels,
            hidden_channels,
            mapping_coeffs,
            internal_weights=False,
            edge_channels_list=edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
            num_experts=num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.so2_conv_2 = SO2Convolution(
            hidden_channels,
            sphere_channels,
            mapping_coeffs,
            num_experts=num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @PrecallInterface.context_handler
    def __call__(
        self,
        node_feats: jax.Array, # In m primary mode
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMatrices,
        edge_envelope: jax.Array,
        *,
        ctx: dict | None = None,
    ):
        num_nodes = len(node_feats)

        messages = jnp.concat((node_feats[senders], node_feats[receivers]), axis=2)

        # Rotate the irreps to align with the edge
        messages = wigner_matrices.rotate(messages)

        # SO2 convolution
        messages, x_0_gating = self.so2_conv_1(messages, edge_embeds, ctx=ctx)
        messages = self.act(x_0_gating, messages)
        messages = self.so2_conv_2(messages, edge_embeds, ctx=ctx)
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
        so3_grid_lmax: SO3Grid,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.so3_grid_lmax = so3_grid_lmax

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
        node_feats_grid = self.so3_grid_lmax.to_grid(node_feats)
        node_feats_grid = self.grid_mlp(node_feats_grid)
        node_feats = self.so3_grid_lmax.from_grid(node_feats_grid)
        return node_feats


class ChargeSpinDatasetEmbed(nnx.Module):
    '''Embeds the charge, spin and dataset information.'''
    def __init__(
        self,
        num_channels: int,
        dataset_size: int | None = None,
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

        self.dataset_size = dataset_size
        if dataset_size is not None:
            self.dataset_emb = nnx.Embed(
                dataset_size, num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

        # Embed + Linear is mathematically equivalent to one_hot + Linear or Embed + add + bais
        self.bais = nnx.Param(initializers.zeros(rngs.params(), (num_channels,), param_dtype))
        self.dtype = dtype

    def __call__(
        self,
        charge: jax.Array,
        spin: jax.Array,
        dataset: jax.Array | None = None,
    ):
        bais, = dtypes.promote_dtype((self.bais.value,), dtype=self.dtype)
        charge_emb = self.charge_emb(charge + 100)
        spin_emb = self.spin_emb(spin)
        embeddings = charge_emb + spin_emb

        if self.dataset_size is not None:
            dataset_emb = self.dataset_emb(dataset)
            embeddings += dataset_emb

        embeddings += bais
        return nnx.silu(embeddings)
