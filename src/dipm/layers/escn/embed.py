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

from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers
import jax
import jax.numpy as jnp

from dipm.layers import MultiLayerPerceptron
from dipm.layers.escn import (
    WignerMatrices,
    MappingCoefficients,
)


class EdgeDegreeEmbedding(nnx.Module):
    """

    Args:
        sphere_channels (int): Number of spherical channels
        mapping_coeffs (MappingCoefficients): Coefficients to convert l and m indices
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example, 
            [input_channels, hidden_channels, hidden_channels]. The last one will be used as hidden
            size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance
            for edge scalar features
        num_species (int): Maximum number of atomic numbers
        rescale_factor (float): Rescale the sum aggregation
    """

    def __init__(
        self,
        sphere_channels: int,
        mapping_coeffs: MappingCoefficients,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool = False,
        num_species: int | None = None,
        rescale_factor: float = 5.0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.lmax = mapping_coeffs.lmax
        self.mmax = mapping_coeffs.mmax
        self.sphere_channels = sphere_channels

        self.m_0_num_coefficients = mapping_coeffs.m_size[0]
        self.m_all_num_coefficents = mapping_coeffs.num_coefficients

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        edge_channels_list = edge_channels_list.copy()
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            assert num_species is not None, "num_species must be provided"
            self.senders_embedding = nnx.Embed(
                num_species, edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001), # Why not xavier?
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            self.receivers_embedding = nnx.Embed(
                num_species, edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001),
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            edge_channels_list[0] = (
                edge_channels_list[0] + 2 * edge_channels_list[-1]
            )
        else:
            self.senders_embedding, self.receivers_embedding = None, None

        # Embedding function of distance
        edge_channels_list.append(
            self.m_0_num_coefficients * self.sphere_channels
        )
        # Radial function
        self.rad_func = MultiLayerPerceptron(
            edge_channels_list,
            activation="silu",
            use_layer_norm=True,
            gradient_normalization=0.0,
            use_bias=True,
            use_act_norm=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.rescale_factor = rescale_factor

    def __call__(
        self,
        node_species: jax.Array,
        edge_distances: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMatrices,
        edge_envelope: jax.Array | None = None,
    ) -> jax.Array:
        num_nodes = node_species.shape[0]

        if self.use_atom_edge_embedding:
            senders_species = node_species[senders]  # Source atom atomic number
            receivers_species = node_species[receivers]  # Target atom atomic number
            senders_embeds = self.senders_embedding(senders_species)
            receivers_embeds = self.receivers_embedding(receivers_species)
            edge_attr = jnp.concat(
                (edge_distances, senders_embeds, receivers_embeds), axis=1
            )
        else:
            edge_attr = edge_distances

        edge_feats_m_0 = self.rad_func(edge_attr)
        edge_feats_m_0 = edge_feats_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        edge_feats_m_pad = jnp.zeros(
            (
                edge_attr.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            dtype=edge_feats_m_0.dtype,
        )
        # edge_feats: [n_edges, (lmax + 1) ^ 2, num_channels], m primary
        edge_feats = jnp.concat((edge_feats_m_0, edge_feats_m_pad), axis=1)

        # Rotate back the irreps
        edge_feats = wigner_matrices.rotate_inv(edge_feats)
        if edge_envelope is not None:
            edge_feats = edge_feats * edge_envelope

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(edge_feats, receivers, num_nodes)
        node_feats = node_feats / self.rescale_factor

        return node_feats
