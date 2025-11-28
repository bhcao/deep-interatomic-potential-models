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

import re

from flax import nnx
from flax.typing import Dtype
import jax
import jax.numpy as jnp

from dipm.layers import (
    get_rbf_cls,
    GraphDropPath,
)
from dipm.layers.escn import (
    SO3Rotation,
    WignerMatrices,
    SO3Grid,
    SO3LinearV2,
    MappingCoefficients,
    EdgeDegreeEmbedding,
    mapping_coefficients,
    get_layernorm_layer,
)
from dipm.data.dataset_info import DatasetInfo
from dipm.models.equiformer_v2.utils import (
    AttntionActivationType,
    FeedForwardType,
)
from dipm.models.force_model import ForceModel
from dipm.models.atomic_energies import get_atomic_energies
from dipm.models.equiformer_v2.config import EquiformerV2Config
from dipm.models.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
)
from dipm.utils.safe_norm import safe_norm


class EquiformerV2(ForceModel):
    """The EquiformerV2 model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Yi-Lun Liao, Brandon Wood, Abhishek Das and Tess Smidt. EquiformerV2:
          Improved Equivariant Transformer for Scaling to Higher-Degree 
          Representations. International Conference on Learning Representations (ICLR),
          January 2024. URL: https://openreview.net/forum?id=mCOBKZmrzD.

    Attributes:
        config: Hyperparameters / configuration for the EquiformerV2 model, see
                :class:`~dipm.models.equiformer_v2.config.EquiformerV2Config`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = EquiformerV2Config
    config: EquiformerV2Config
    force_head_prefix = "equiformer_model.force_block"
    embedding_layer_regexp = re.compile(r"\.(sphere|senders|receivers)_embedding\.embedding$")

    def __init__(
        self,
        config: dict | EquiformerV2Config,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(42)
        super().__init__(config, dataset_info, dtype=dtype)

        r_max = self.dataset_info.cutoff_distance_angstrom

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

        avg_num_nodes = self.config.avg_num_nodes
        if avg_num_nodes is None:
            avg_num_nodes = self.dataset_info.avg_num_nodes

        num_species = len(self.dataset_info.atomic_energies_map)

        # Decide the feedforward network type
        if self.config.use_grid_mlp:
            ff_type = FeedForwardType.GRID
            if self.config.use_sep_s2_act:
                ff_type = FeedForwardType.GRID_SEP
        elif self.config.use_gate_act:
            ff_type = FeedForwardType.GATE
        elif self.config.use_sep_s2_act:
            ff_type = FeedForwardType.S2_SEP
        else:
            ff_type = FeedForwardType.S2

        # Decide the attention activation type
        if self.config.use_gate_act:
            attn_act_type = AttntionActivationType.GATE
        elif self.config.use_sep_s2_act:
            attn_act_type = AttntionActivationType.S2_SEP
        else:
            attn_act_type = AttntionActivationType.S2

        equiformer_kargs = dict(
            avg_num_neighbors=avg_num_neighbors,
            num_layers=self.config.num_layers,
            lmax=self.config.lmax,
            mmax=self.config.mmax,
            sphere_channels=self.config.sphere_channels,
            num_edge_channels=self.config.num_edge_channels,
            atom_edge_embedding=self.config.atom_edge_embedding,
            num_rbf=self.config.num_rbf,
            attn_hidden_channels=self.config.attn_hidden_channels,
            num_heads=self.config.num_heads,
            attn_alpha_channels=self.config.attn_alpha_channels,
            attn_value_channels=self.config.attn_value_channels,
            ffn_hidden_channels=self.config.ffn_hidden_channels,
            norm_type=self.config.norm_type,
            grid_resolution=self.config.grid_resolution,
            use_m_share_rad=self.config.use_m_share_rad,
            use_attn_renorm=self.config.use_attn_renorm,
            attn_act_type=attn_act_type,
            ff_type=ff_type,
            alpha_drop=self.config.alpha_drop,
            drop_path_rate=self.config.drop_path_rate,
            avg_num_nodes=avg_num_nodes,
            rbf_type="gauss",
            trainable_rbf=False,
            rbf_width=2.0,
            cutoff=r_max,
            num_species=num_species,
            predict_forces=self.config.force_head,
        )

        self.equiformer_model = EquiformerV2Block(
            **equiformer_kargs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.atomic_energies = nnx.Cache(get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species, dtype=self.dtype
        ))

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        n_node: jax.Array, # Nel version of pyg.Data.batch
        rngs: nnx.Rngs | None = None, # Rngs for dropout, None for eval
    ) -> jax.Array:
        node_energies = self.equiformer_model(
            edge_vectors, node_species, senders, receivers, n_node, rngs
        )

        if self.config.force_head:
            node_energies, forces = node_energies

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        node_energies += self.atomic_energies.value[node_species]  # [n_nodes, ]

        if self.config.force_head:
            return node_energies, std * forces
        return node_energies


class EquiformerV2Block(nnx.Module):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon
    S2 activation.
    """

    def __init__(
        self,
        avg_num_neighbors: float,
        num_layers: int,
        lmax: int,
        mmax: int,
        sphere_channels: int,
        num_species: int,
        num_edge_channels: int,
        atom_edge_embedding: str,
        attn_hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        norm_type: str,
        grid_resolution: int,
        use_m_share_rad: bool,
        use_attn_renorm: bool,
        attn_act_type: AttntionActivationType,
        ff_type: FeedForwardType,
        alpha_drop: float,
        drop_path_rate: float,
        avg_num_nodes: float,
        num_rbf: int = 600,
        rbf_type: str = "gauss",
        trainable_rbf: bool = False,
        rbf_width: float = 2.0,
        cutoff: float = 5.0,
        predict_forces: bool = False,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        self.lmax = lmax
        self.mmax = mmax
        self.avg_num_nodes = avg_num_nodes
        self.deterministic = False # Randomness

        # Weights for message initialization
        self.sphere_embedding = nnx.Embed(
            num_species, sphere_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # Function used to measure the distances between atoms
        self.distance_expansion = get_rbf_cls(rbf_type)(
            cutoff,
            num_rbf,
            trainable=trainable_rbf,
            rbf_width=rbf_width,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # Sizes of radial functions (input channels and 2 hidden channels)
        edge_channels_list = [num_rbf] + [num_edge_channels] * 2

        # Atom edge embedding
        self.senders_embedding, self.receivers_embedding = None, None
        if atom_edge_embedding == 'shared':
            self.senders_embedding = nnx.Embed(
                num_species, num_edge_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            self.receivers_embedding = nnx.Embed(
                num_species, num_edge_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            edge_channels_list[0] += 2 * num_edge_channels

        # Initialize conversion between degree l and order m layouts
        mapping_coeffs = mapping_coefficients(lmax, mmax)

        # Initialize the transformations between spherical and grid representations
        so3_grid = SO3Grid(lmax, mmax, resolution=grid_resolution, dtype=dtype or param_dtype)
        so3_grid_lmax = SO3Grid(
            lmax, lmax, resolution=grid_resolution, dtype=dtype or param_dtype
        )

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels,
            mapping_coeffs,
            edge_channels_list,
            atom_edge_embedding == 'isolated',
            num_species=num_species,
            rescale_factor=avg_num_neighbors,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.layers = []
        for _ in range(num_layers):
            layer = EquiformerV2Layer(
                sphere_channels,
                attn_hidden_channels,
                num_heads,
                attn_alpha_channels,
                attn_value_channels,
                ffn_hidden_channels,
                sphere_channels,
                mapping_coeffs,
                so3_grid,
                so3_grid_lmax,
                num_species,
                edge_channels_list,
                atom_edge_embedding == 'isolated',
                use_m_share_rad,
                use_attn_renorm,
                attn_act_type,
                ff_type,
                norm_type,
                alpha_drop,
                drop_path_rate,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.layers.append(layer)

        # Output blocks for energy and forces
        self.norm = get_layernorm_layer(
            norm_type,
            lmax=lmax,
            num_channels=sphere_channels,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.energy_block = FeedForwardNetwork(
            sphere_channels,
            ffn_hidden_channels,
            1,
            lmax,
            so3_grid_lmax,
            ff_type,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if predict_forces:
            self.force_block = SO2EquivariantGraphAttention(
                sphere_channels,
                attn_hidden_channels,
                num_heads,
                attn_alpha_channels,
                attn_value_channels,
                1,
                mapping_coeffs,
                so3_grid,
                num_species,
                edge_channels_list,
                atom_edge_embedding == 'isolated',
                use_m_share_rad,
                use_attn_renorm,
                attn_act_type,
                alpha_drop=0.0,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.force_block = None

        self.so3_rotation = SO3Rotation(lmax, mmax, mapping_coeffs.perm, dtype=dtype or param_dtype)

        self.regress_forces = predict_forces

    def __call__(
        self,
        edge_vectors: jax.Array,  # [n_edges, 3]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        n_node: jax.Array,  # [batch_size]
        rngs: nnx.Rngs | None = None,  # Rngs for dropout, None for eval
    ):
        num_atoms = len(node_species)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        if not self.deterministic:
            assert rngs is not None
            rot_gamma = jax.random.uniform(
                rngs['rotation'](), shape=len(edge_vectors), maxval=2 * jnp.pi,
                dtype=edge_vectors.dtype
            )
        else:
            rot_gamma = jnp.zeros(len(edge_vectors), dtype=edge_vectors.dtype)

        wigner_matrices = self.so3_rotation.create_wigner_matrices(edge_vectors, rot_gamma)

        # Initialize the l = 0, m = 0 coefficients
        node_feats_0 = self.sphere_embedding(node_species)[:, None]
        node_feats_m_pad = jnp.zeros(
            [num_atoms, (self.lmax + 1) ** 2 - 1, node_feats_0.shape[-1]],
            dtype=edge_vectors.dtype,
        )
        node_feats = jnp.concat((node_feats_0, node_feats_m_pad), axis=1)

        # Edge encoding (distance and atom edge)
        edge_distances = safe_norm(edge_vectors, axis=-1)
        edge_distances = self.distance_expansion(edge_distances)
        if self.senders_embedding is not None:
            senders_species = node_species[senders] # Source atom atomic number
            target_species = node_species[receivers] # Target atom atomic number
            senders_embedding = self.senders_embedding(senders_species)
            receivers_embedding = self.receivers_embedding(target_species)
            edge_distances = jnp.concat(
                (edge_distances, senders_embedding, receivers_embedding), axis=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            node_species, edge_distances, senders, receivers, wigner_matrices
        )
        node_feats = node_feats + edge_degree

        for layer in self.layers:
            node_feats = layer(
                node_feats,
                node_species,
                edge_distances,
                senders,
                receivers,
                wigner_matrices,
                n_node=n_node,  # for GraphDropPath
                rngs=rngs,
            )

        # Final layer norm
        node_feats = self.norm(node_feats)

        node_energies = self.energy_block(node_feats)
        node_energies = node_energies[:, 0, 0] / self.avg_num_nodes

        if self.regress_forces:
            forces = self.force_block(
                node_feats,
                node_species,
                edge_distances,
                senders,
                receivers,
                wigner_matrices,
                rngs=rngs,
            )
            forces = forces[:, 1:4, 0]
            return node_energies, forces

        return node_energies


class EquiformerV2Layer(nnx.Module):
    """

    Args:
        sphere_channels (int): Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int): Number of attention heads
        attn_alpha_channels (int): Number of channels for alpha vector in each attention head
        attn_value_channels (int): Number of channels for value vector in each attention head
        ffn_hidden_channels (int): Number of hidden channels used during feedforward network
        output_channels (int): Number of output channels
        mapping_coeffs (MappingCoefficients): Coefficients to convert l and m indices
        so3_grid (SO3Grid): Class used to convert between grid and the spherical harmonic
        so3_grid_lmax (SO3Grid): Class used to convert between grid and the mmax=lmax spherical
            harmonic
        num_species (int): Maximum number of atomic numbers
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example,
            [input_channels, hidden_channels, hidden_channels]. The last one will be used as hidden
            size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative
            distance for edge scalar features
        use_m_share_rad (bool): Whether all m components within a type-L vector of one channel
            share radial function weights
        use_attn_renorm (bool): Whether to re-normalize attention weights
        attn_act_type (AttntionActivationType): Type of activation function used for attention
        ff_type (FeedForwardType): Type of feedforward network used
        norm_type (str): Type of normalization layer (['layer_norm', 'layer_norm_sh'])
        alpha_drop (float): Dropout rate for attention weights
        drop_path_rate (float): Drop path rate
    """

    def __init__(
        self,
        sphere_channels: int,
        attn_hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        output_channels: int,
        mapping_coeffs: MappingCoefficients,
        so3_grid: SO3Grid,
        so3_grid_lmax: SO3Grid,
        num_species: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_attn_renorm: bool = True,
        attn_act_type: AttntionActivationType = AttntionActivationType.S2_SEP,
        ff_type: FeedForwardType = FeedForwardType.GRID_SEP,
        norm_type: str = "rms_norm_sh",
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.norm_1 = get_layernorm_layer(
            norm_type, lmax=mapping_coeffs.lmax, num_channels=sphere_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        self.graph_attn = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            mapping_coeffs=mapping_coeffs,
            so3_grid=so3_grid,
            num_species=num_species,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            use_attn_renorm=use_attn_renorm,
            attn_act_type=attn_act_type,
            alpha_drop=alpha_drop,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.drop_path = (
            GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        )

        self.norm_2 = get_layernorm_layer(
            norm_type, lmax=mapping_coeffs.lmax, num_channels=sphere_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels,
            output_channels=output_channels,
            lmax=mapping_coeffs.lmax,
            so3_grid_lmax=so3_grid_lmax,
            ff_type=ff_type,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3LinearV2(
                sphere_channels, output_channels, lmax=mapping_coeffs.lmax,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
        else:
            self.ffn_shortcut = None

    def __call__(
        self,
        node_feats: jax.Array,
        node_species: jax.Array,
        edge_distances: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMatrices,
        n_node: jax.Array,  # for GraphDropPath
        rngs: nnx.Rngs | None = None,
    ):
        # Attention block
        node_feats_res = node_feats
        node_feats = self.norm_1(node_feats)
        node_feats = self.graph_attn(
            node_feats, node_species, edge_distances, senders, receivers, wigner_matrices, rngs
        )

        if self.drop_path is not None:
            node_feats = self.drop_path(node_feats, n_node, rngs=rngs)

        node_feats = node_feats + node_feats_res

        # FFN block
        node_feats_res = node_feats
        node_feats = self.norm_2(node_feats)
        node_feats = self.ffn(node_feats)

        if self.drop_path is not None:
            node_feats = self.drop_path(node_feats, n_node, rngs=rngs)

        if self.ffn_shortcut is not None:
            node_feats_res = self.ffn_shortcut(node_feats_res)

        node_feats = node_feats + node_feats_res

        return node_feats
