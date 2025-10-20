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

import copy

from flax import nnx
from flax.typing import Dtype, Initializer
from flax.nnx.nn import initializers
import jax
import jax.numpy as jnp

from dipm.layers import (
    MultiLayerPerceptron,
    SO3Rotation,
    expand_index,
)
from dipm.models.equiformer_v2.utils import MappingCoefficients


class SO3LinearV2(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax: int,
        *,
        kernel_init: Initializer = initializers.lecun_normal(),
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """
        1. Use `jnp.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        key = rngs.params()
        self.weight = nnx.Param(
            kernel_init(
                key, ((self.lmax + 1), in_features, out_features), param_dtype
            )
        )
        key = rngs.params()
        self.bias = nnx.Param(initializers.zeros(key, out_features, param_dtype))

        self.expand_index = expand_index(self.lmax)

    def __call__(self, embedding):
        weight = self.weight.value[self.expand_index] # [(L_max + 1) ** 2, C_in, C_out]
        out = jnp.einsum(
            "bmi, mio -> bmo", embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        out = out.at[:, 0:1, :].add(
            self.bias.value.reshape(1, 1, self.out_features)
        )

        return out


class EdgeDegreeEmbedding(nnx.Module):
    """

    Args:
        sphere_channels (int): Number of spherical channels
        lmax (int): Maximum degrees (l) for each resolution
        mmax (int): Maximum orders (m) for each resolution
        so3_rotation (SO3Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        to_m (jnp.ndarray): Array to convert l and m indices once node embedding is rotated
        m_size (jnp.ndarray): Array to store the number of coefficients for each degree (l)
        num_species (int): Maximum number of atomic numbers
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example, 
            [input_channels, hidden_channels, hidden_channels]. The last one will be used as hidden
            size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance
            for edge scalar features
        rescale_factor (float): Rescale the sum aggregation
    """

    def __init__(
        self,
        sphere_channels: int,
        so3_rotation: SO3Rotation,
        mapping_coeffs: MappingCoefficients,
        num_species: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool,
        rescale_factor: float,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.lmax = mapping_coeffs.lmax
        self.mmax = mapping_coeffs.mmax
        self.sphere_channels = sphere_channels
        self.so3_rotation = so3_rotation
        self.to_m = mapping_coeffs.to_m

        self.m_0_num_coefficients = mapping_coeffs.m_size[0]
        self.m_all_num_coefficents = mapping_coeffs.num_coefficients

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        edge_channels_list = edge_channels_list.copy()
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            self.source_embedding = nnx.Embed(
                num_species, edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001), # Why not xavier?
                param_dtype=param_dtype, rngs=rngs,
            )
            self.target_embedding = nnx.Embed(
                num_species, edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001),
                param_dtype=param_dtype, rngs=rngs,
            )
            edge_channels_list[0] = (
                edge_channels_list[0] + 2 * edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

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
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.rescale_factor = rescale_factor

    def __call__(
        self,
        node_species: jnp.ndarray,
        edge_distances: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> jnp.ndarray:
        num_nodes = node_species.shape[0]

        if self.use_atom_edge_embedding:
            source_element = node_species[senders]  # Source atom atomic number
            target_element = node_species[receivers]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_attr = jnp.concat(
                (edge_distances, source_embedding, target_embedding), axis=1
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
        )
        # edge_feats: [n_edges, (lmax + 1) ^ 2, num_channels]
        edge_feats = jnp.concat((edge_feats_m_0, edge_feats_m_pad), axis=1)

        # Reshape the spherical harmonics based on l (degree)
        edge_feats = jnp.einsum("nac, ab -> nbc", edge_feats, self.to_m)

        # Rotate back the irreps
        edge_feats = self.so3_rotation.rotate_inv(edge_feats)

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(edge_feats, receivers, num_nodes)
        node_feats = node_feats / self.rescale_factor

        return node_feats


class SO2mConvolution(nnx.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int): Order of the spherical harmonic coefficients
        sphere_channels (int): Number of spherical channels
        m_output_channels (int): Number of output channels used during the SO(2) conv
        lmax (int): Degrees (l)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.fc = nnx.Linear(
            (lmax - m + 1) * sphere_channels,
            2 * (lmax - m + 1) * m_output_channels,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, edge_feats_m):
        edge_feats_m = self.fc(edge_feats_m)
        edge_feats_r, edge_feats_i = jnp.split(edge_feats_m, 2, axis=2)
        edge_feats_m_r = edge_feats_r[:, 0] - edge_feats_i[:, 1]
        edge_feats_m_i = edge_feats_r[:, 1] + edge_feats_i[:, 0]
        edge_feats_m = jnp.stack((edge_feats_m_r, edge_feats_m_i), axis=1)

        return edge_feats_m


class SO2Convolution(nnx.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int): Number of spherical channels
        m_output_channels (int): Number of output channels used during the SO(2) conv
        lmax (int): Max degree (l)
        mmax (int): Max order (m)
        to_m (jnp.ndarray): Used to extract a subset of m components
        m_size (jnp.ndarray): Size of each degree
        internal_weights (bool): If True, not using radial function to multiply inputs features
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example,
            [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out` and `extra_m0_features`.
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        mapping_coeffs: MappingCoefficients,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.m_output_channels = m_output_channels
        self.lmax = mapping_coeffs.lmax
        self.mmax = mapping_coeffs.mmax
        self.to_m = mapping_coeffs.to_m
        self.m_size = mapping_coeffs.m_size
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_m0 = (self.lmax + 1) * sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = (self.lmax + 1) * m_output_channels

        if extra_m0_output_channels is not None:
            m0_output_channels = (
                m0_output_channels + extra_m0_output_channels
            )
        self.fc_m0 = nnx.Linear(
            num_channels_m0, m0_output_channels, param_dtype=param_dtype, rngs=rngs
        )
        num_channels_rad = self.fc_m0.in_features # for radial function

        # SO(2) convolution for non-zero m
        self.so2_m_conv = []
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2mConvolution(
                    m,
                    sphere_channels,
                    m_output_channels,
                    self.lmax,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
            num_channels_rad = (
                num_channels_rad + self.so2_m_conv[-1].fc.in_features
            )

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            if self.edge_channels_list is None:
                raise ValueError(
                    "If `internal_weights` is False, `edge_channels_list` must be provided."
                )
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = MultiLayerPerceptron(
                self.edge_channels_list,
                activation="silu",
                use_layer_norm=True,
                gradient_normalization=0.0,
                use_bias=True,
                use_act_norm=False,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(
        self,
        edge_feats: jnp.ndarray,
        edge_embeds: jnp.ndarray
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        num_edges = len(edge_embeds)

        # Reshape the spherical harmonics based on m (order)
        edge_feats = jnp.einsum("nac, ba -> nbc", edge_feats, self.to_m)

        # radial function
        if self.rad_func is not None:
            edge_embeds = self.rad_func(edge_embeds)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        edge_feats_0 = edge_feats[:, :self.m_size[0]]
        edge_feats_0 = edge_feats_0.reshape(num_edges, -1)
        if self.rad_func is not None:
            edge_embeds_0 = edge_embeds[:, :self.fc_m0.in_features]
            edge_feats_0 = edge_feats_0 * edge_embeds_0
        edge_feats_0 = self.fc_m0(edge_feats_0)

        edge_feats_0_extra = None
        # extract extra m0 features
        if self.extra_m0_output_channels is not None:
            edge_feats_0_extra = edge_feats_0[..., :self.extra_m0_output_channels]
            edge_feats_0 = edge_feats_0[..., self.extra_m0_output_channels:self.fc_m0.out_features]

        edge_feats_0 = edge_feats_0.reshape(num_edges, -1, self.m_output_channels)
        # x[:, 0 : self.mappingReduced.m_size[0]] = edge_feats_0
        edge_feats_out = [edge_feats_0]
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients
            edge_feats_m = edge_feats[:, offset : 2*self.m_size[m]+offset]
            edge_feats_m = edge_feats_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            if self.rad_func is not None:
                edge_embeds_m = edge_embeds[
                    :,
                    offset_rad : self.so2_m_conv[m - 1].fc.in_features + offset_rad
                ]
                edge_embeds_m = edge_embeds_m.reshape(
                    num_edges, 1, self.so2_m_conv[m - 1].fc.in_features
                )
                edge_feats_m = edge_feats_m * edge_embeds_m
            edge_feats_m = self.so2_m_conv[m - 1](edge_feats_m)
            edge_feats_m = edge_feats_m.reshape(num_edges, -1, self.m_output_channels)
            # x[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = edge_feats_m
            edge_feats_out.append(edge_feats_m)
            offset = offset + 2 * self.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

        edge_feats = jnp.concat(edge_feats_out, axis=1)
        # Reshape the spherical harmonics based on l (degree)
        edge_feats = jnp.einsum("nac, ab -> nbc", edge_feats, self.to_m)

        if self.extra_m0_output_channels is not None:
            return edge_feats, edge_feats_0_extra
        return edge_feats
