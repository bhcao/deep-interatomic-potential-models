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

from dipm.layers import (
    MultiLayerPerceptron,
    SO3Rotation,
    SO3Grid,
    expand_index,
)
from dipm.layers.activations import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from dipm.models.equiformer_v2.blocks import (
    SO3LinearV2,
    SO2Convolution,
)
from dipm.models.equiformer_v2.utils import (
    MappingCoefficients,
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
        so3_rotation (SO3Rotation): Class to calculate Wigner-D matrices and rotate embeddings
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
        use_gate_act (bool): If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool): If `True`, use separable S2 activation when `use_gate_act` is False.
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
        so3_rotation: SO3Rotation,
        mapping_coeffs: MappingCoefficients,
        so3_grid: SO3Grid,
        num_species: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = False,
        use_sep_s2_act: bool = True,
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
        self.so3_rotation = so3_rotation

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        edge_channels_list = edge_channels_list.copy()
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

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
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        # Create SO(2) convolution blocks
        extra_m0_output_channels = num_heads * attn_alpha_channels
        if self.use_gate_act:
            extra_m0_output_channels = extra_m0_output_channels + self.lmax * hidden_channels
        else:
            if self.use_sep_s2_act:
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

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                self.s2_act = SeparableS2Activation(so3_grid)
            else:
                self.s2_act = S2Activation(so3_grid)

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
        node_feats: jnp.ndarray,
        node_species: jnp.ndarray,
        edge_distances: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
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

        edge_feats = jnp.concat(
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
            edge_feats = edge_feats * edge_embeds_weight

        # Rotate the irreps to align with the edge
        edge_feats = self.so3_rotation.rotate(edge_feats)

        # First SO(2)-convolution
        edge_feats, x_0_extra = self.so2_conv_1(edge_feats, edge_embeds)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra[:, x_alpha_num_channels:x_0_extra.shape[1]] # for activation
            x_0_alpha = x_0_extra[:, :x_alpha_num_channels] # for attention weights
            edge_feats = self.gate_act(x_0_gating, edge_feats)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra[:, x_alpha_num_channels:x_0_extra.shape[1]] # for activation
                x_0_alpha = x_0_extra[:, :x_alpha_num_channels] # for attention weights
                edge_feats = self.s2_act(x_0_gating, edge_feats)
            else:
                x_0_alpha = x_0_extra
                edge_feats = self.s2_act(edge_feats)
            # x_message._grid_act(self.so3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        edge_feats = self.so2_conv_2(edge_feats, edge_embeds)

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
        attn = edge_feats.reshape(
            edge_feats.shape[0],
            edge_feats.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )

        # Rotate back the irreps
        edge_feats = self.so3_rotation.rotate_inv(attn)

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(edge_feats, receivers, num_nodes)

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
        use_gate_act (bool): If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool): If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool): If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        output_channels: int,
        lmax: int,
        so3_grid: SO3Grid,
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.so3_grid = so3_grid

        sphere_channels_all = sphere_channels

        self.so3_linear_1 = SO3LinearV2(
            sphere_channels_all, hidden_channels, lmax=lmax,
            param_dtype=param_dtype, rngs=rngs
        )

        self.so3_linear_2 = SO3LinearV2(
            hidden_channels, output_channels, lmax=lmax,
            param_dtype=param_dtype, rngs=rngs
        )

        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nnx.Sequential(
                    nnx.Linear(
                        sphere_channels_all,
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

        if self.use_gate_act:
            self.gating_linear = nnx.Linear(
                sphere_channels_all,
                lmax * hidden_channels,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.gate_act = GateActivation(lmax, lmax, hidden_channels)
        else:
            if self.use_sep_s2_act:
                self.gating_linear = nnx.Linear(
                    sphere_channels_all, hidden_channels,
                    param_dtype=param_dtype, rngs=rngs
                )
                self.s2_act = SeparableS2Activation(self.so3_grid)
            else:
                self.gating_linear = None
                self.s2_act = S2Activation(self.so3_grid)


    def __call__(self, node_feats: jnp.ndarray):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(node_feats[:, 0:1])
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(node_feats[:, 0:1])

        node_feats = self.so3_linear_1(node_feats)

        if self.use_grid_mlp:
            # Project to grid
            node_feats_grid = self.so3_grid.to_grid(node_feats)
            # Perform point-wise operations
            node_feats_grid = self.grid_mlp(node_feats_grid)
            # Project back to spherical harmonic coefficients
            node_feats = self.so3_grid.from_grid(node_feats_grid)

            if self.use_sep_s2_act:
                node_feats = jnp.concat(
                    (gating_scalars, node_feats[:, 1:]),
                    axis=1,
                )
        else:
            if self.use_gate_act:
                node_feats = self.gate_act(gating_scalars, node_feats)
            else:
                if self.use_sep_s2_act:
                    node_feats = self.s2_act(gating_scalars, node_feats)
                else:
                    node_feats = self.s2_act(node_feats)

        node_feats = self.so3_linear_2(node_feats)

        return node_feats
