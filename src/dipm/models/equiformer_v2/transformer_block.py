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
from flax.typing import Dtype
from flax.nnx.nn import initializers
import jax
import jax.numpy as jnp

from dipm.layers import MultiLayerPerceptron
from dipm.layers.escn import (
    WignerMatrices,
    SO3Grid,
    SO3LinearV2,
    expand_index,
    MappingCoefficients,
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SO2Convolution,
)
from dipm.layers.activations import (
    SmoothLeakyReLU,
)
from dipm.models.equiformer_v2.utils import (
    AttntionActivationType,
    FeedForwardType,
    pyg_softmax,
)


class SO2EquivariantGraphAttention(nnx.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention
        weights and non-linear messages attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int): Number of spherical channels
        hidden_channels (int): Number of hidden channels used during the SO(2) conv
        num_heads (int): Number of attention heads
        attn_alpha_channels (int): Number of channels for alpha vector in each attention head
        attn_value_channels (int): Number of channels for value vector in each attention head
        output_channels (int): Number of output channels
        mapping_coeffs (MappingCoefficients): Data for converting indices and max degeree/order
        so3_grid (SO3Grid): Class used to convert from grid the spherical harmonic representations
        num_species (int): Maximum number of atomic numbers
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example, 
            [input_channels, hidden_channels, hidden_channels]. The last one will be used as hidden
            size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative
            distance for edge scalar features
        use_m_share_rad (bool): Whether all m components within a type-L vector of one channel
            share radial function weights
        use_attn_renorm (bool): Whether to re-normalize attention weights
        attn_act_type (AttntionActivationType): Type of attention activation function
        alpha_drop (float): Dropout rate for attention weights
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        output_channels: int,
        mapping_coeffs: MappingCoefficients,
        so3_grid: SO3Grid,
        num_species: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_attn_renorm: bool = True,
        attn_act_type: AttntionActivationType = AttntionActivationType.S2_SEP,
        alpha_drop: float = 0.0,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.sphere_channels = sphere_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.lmax = mapping_coeffs.lmax
        self.mmax = mapping_coeffs.mmax
        self.perm = mapping_coeffs.perm

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        edge_channels_list = edge_channels_list.copy()
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.attn_act_type = attn_act_type

        if self.use_atom_edge_embedding:
            self.senders_embedding = nnx.Embed(
                num_species, edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001), # Why not xavier?
                param_dtype=param_dtype, rngs=rngs,
            )
            self.receivers_embedding = nnx.Embed(
                num_species, edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001),
                param_dtype=param_dtype, rngs=rngs,
            )
            edge_channels_list[0] = (
                edge_channels_list[0] + 2 * edge_channels_list[-1]
            )
        else:
            self.senders_embedding, self.receivers_embedding = None, None

        self.use_attn_renorm = use_attn_renorm

        # Create SO(2) convolution blocks
        extra_m0_output_channels = num_heads * attn_alpha_channels
        if attn_act_type == AttntionActivationType.GATE:
            extra_m0_output_channels = extra_m0_output_channels + self.lmax * hidden_channels
        elif attn_act_type == AttntionActivationType.S2_SEP:
            extra_m0_output_channels = extra_m0_output_channels + hidden_channels

        if use_m_share_rad:
            edge_channels_list = edge_channels_list + [
                2 * sphere_channels * (self.lmax + 1)
            ]
            self.rad_func = MultiLayerPerceptron(
                edge_channels_list,
                activation="silu",
                use_layer_norm=True,
                gradient_normalization=0.0,
                use_bias=True,
                use_act_norm=False,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.expand_index = expand_index(self.lmax)

        self.so2_conv_1 = SO2Convolution(
            2 * sphere_channels,
            hidden_channels,
            mapping_coeffs,
            internal_weights=use_m_share_rad,
            edge_channels_list=(
                edge_channels_list if not use_m_share_rad else None
            ),
            # for attention weights and/or gate activation
            extra_m0_output_channels=extra_m0_output_channels,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if self.use_attn_renorm:
            self.alpha_norm = nnx.LayerNorm(
                attn_alpha_channels,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.alpha_norm = lambda x: x
        self.alpha_act = SmoothLeakyReLU()
        key = rngs.params()
        self.alpha_dot = nnx.Param(
            initializers.lecun_normal(in_axis=-1, out_axis=-2)(
                key, (num_heads, attn_alpha_channels), param_dtype
            )
        )
        # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = nnx.Dropout(alpha_drop)

        if attn_act_type == AttntionActivationType.GATE:
            self.gate_act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=hidden_channels,
                m_prime=True,
            )
        elif attn_act_type == AttntionActivationType.S2_SEP:
            self.s2_act = SeparableS2Activation(so3_grid, self.perm)
        else:
            self.s2_act = S2Activation(so3_grid, self.perm)

        self.so2_conv_2 = SO2Convolution(
            hidden_channels,
            num_heads * attn_value_channels,
            mapping_coeffs,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,  # for attention weights
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.proj = SO3LinearV2(
            num_heads * attn_value_channels,
            output_channels,
            lmax=self.lmax,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jax.Array,
        node_species: jax.Array,
        edge_distances: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMatrices,
        rngs: nnx.Rngs | None = None,
    ):
        num_nodes = node_feats.shape[0]
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            senders_species = node_species[senders]  # Source atom atomic number
            receivers_species = node_species[receivers]  # Target atom atomic number
            senders_embeds = self.senders_embedding(senders_species)
            receivers_embeds = self.receivers_embedding(receivers_species)
            edge_embeds = jnp.concat(
                (edge_distances, senders_embeds, receivers_embeds), axis=1
            )
        else:
            edge_embeds = edge_distances

        messages = jnp.concat(
            (node_feats[senders], node_feats[receivers]), axis=2
        )

        # radial function (scale all m components within a type-L vector of one channel
        # with the same weight)
        if self.use_m_share_rad:
            edge_embeds_weight = self.rad_func(edge_embeds)
            edge_embeds_weight = edge_embeds_weight.reshape(
                -1, (self.lmax + 1), 2 * self.sphere_channels
            )
            # [E, (L_max + 1) ** 2, C]
            edge_embeds_weight = edge_embeds_weight[:, self.expand_index]
            messages = messages * edge_embeds_weight

        # Rotate the irreps to align with the edge, get m primary
        messages = wigner_matrices.rotate(messages)

        # First SO(2)-convolution
        messages, x_0_extra = self.so2_conv_1(messages, edge_embeds)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.attn_act_type == AttntionActivationType.GATE:
            # Gate activation
            x_0_gating = x_0_extra[:, x_alpha_num_channels:] # for activation
            x_0_alpha = x_0_extra[:, :x_alpha_num_channels] # for attention weights
            messages = self.gate_act(x_0_gating, messages)
        elif self.attn_act_type == AttntionActivationType.S2_SEP:
            x_0_gating = x_0_extra[:, x_alpha_num_channels:] # for activation
            x_0_alpha = x_0_extra[:, :x_alpha_num_channels] # for attention weights
            messages = self.s2_act(x_0_gating, messages)
        else:
            x_0_alpha = x_0_extra
            messages = self.s2_act(messages)
            # x_message._grid_act(self.so3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        messages = self.so2_conv_2(messages, edge_embeds)

        # Attention weights
        x_0_alpha = x_0_alpha.reshape(
            -1, self.num_heads, self.attn_alpha_channels
        )
        x_0_alpha = self.alpha_norm(x_0_alpha)
        x_0_alpha = self.alpha_act(x_0_alpha)
        alpha = jnp.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot.value)
        alpha = pyg_softmax(alpha, receivers, num_nodes)
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha, rngs=rngs)

        # Attention weights * non-linear messages
        attn = messages.reshape(
            messages.shape[0],
            messages.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        messages = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )

        # Rotate back the irreps
        messages = wigner_matrices.rotate_inv(messages)

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(messages, receivers, num_nodes)

        # Project
        node_feats = self.proj(node_feats)

        return node_feats


class FeedForwardNetwork(nnx.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int): Number of spherical channels
        hidden_channels (int): Number of hidden channels used during feedforward network
        output_channels (int): Number of output channels
        lmax (int): Degree (l)
        so3_grid (SO3Grid): Class used to convert from grid the spherical harmonic representations
        ff_type (FeedForwardType): Type of feedforward network
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        output_channels: int,
        lmax: int,
        so3_grid_lmax: SO3Grid,
        ff_type: FeedForwardType,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.ff_type = ff_type
        self.so3_grid_lmax = so3_grid_lmax

        self.so3_linear_1 = SO3LinearV2(
            sphere_channels, hidden_channels, lmax=lmax,
            param_dtype=param_dtype, rngs=rngs
        )

        self.so3_linear_2 = SO3LinearV2(
            hidden_channels, output_channels, lmax=lmax,
            param_dtype=param_dtype, rngs=rngs
        )

        if ff_type in [FeedForwardType.GRID, FeedForwardType.GRID_SEP]:
            if ff_type == FeedForwardType.GRID_SEP:
                self.scalar_mlp = nnx.Sequential(
                    nnx.Linear(
                        sphere_channels,
                        hidden_channels,
                        param_dtype=param_dtype,
                        rngs=rngs,
                    ),
                    nnx.silu,
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nnx.Sequential(
                nnx.Linear(
                    hidden_channels, hidden_channels, use_bias=False,
                    param_dtype=param_dtype, rngs=rngs
                ),
                nnx.silu,
                nnx.Linear(
                    hidden_channels, hidden_channels, use_bias=False,
                    param_dtype=param_dtype, rngs=rngs
                ),
                nnx.silu,
                nnx.Linear(
                    hidden_channels, hidden_channels, use_bias=False,
                    param_dtype=param_dtype, rngs=rngs
                ),
            )
            return

        if ff_type == FeedForwardType.GATE:
            self.gating_linear = nnx.Linear(
                sphere_channels,
                lmax * hidden_channels,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.gate_act = GateActivation(lmax, lmax, hidden_channels)
        if ff_type == FeedForwardType.S2_SEP:
            self.gating_linear = nnx.Linear(
                sphere_channels, hidden_channels, param_dtype=param_dtype, rngs=rngs
            )
            self.s2_act = SeparableS2Activation(so3_grid_lmax)
        else:
            self.s2_act = S2Activation(so3_grid_lmax)


    def __call__(self, node_feats: jax.Array):
        node_feats_orig = node_feats
        node_feats = self.so3_linear_1(node_feats)

        if self.ff_type in [FeedForwardType.GRID, FeedForwardType.GRID_SEP]:
            node_feats_grid = self.so3_grid_lmax.to_grid(node_feats)
            node_feats_grid = self.grid_mlp(node_feats_grid)
            node_feats = self.so3_grid_lmax.from_grid(node_feats_grid)

            if self.ff_type == FeedForwardType.GRID_SEP:
                gating_scalars = self.scalar_mlp(node_feats_orig[:, 0:1])
                node_feats = jnp.concat(
                    (gating_scalars, node_feats[:, 1:]),
                    axis=1,
                )
        elif self.ff_type == FeedForwardType.GATE:
            gating_scalars = self.gating_linear(node_feats_orig[:, 0:1])
            node_feats = self.gate_act(gating_scalars, node_feats)
        elif self.ff_type == FeedForwardType.S2_SEP:
            gating_scalars = self.gating_linear(node_feats_orig[:, 0:1])
            node_feats = self.s2_act(gating_scalars, node_feats)
        else:
            node_feats = self.s2_act(node_feats)

        node_feats = self.so3_linear_2(node_feats)

        return node_feats
