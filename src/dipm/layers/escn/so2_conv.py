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

import copy

from flax import nnx
from flax.typing import Dtype
import jax
import jax.numpy as jnp

from dipm.layers import MultiLayerPerceptron
from dipm.layers.escn.utils import MappingCoefficients
from dipm.layers.escn.linear import MoLE
from dipm.models.force_model import PrecallInterface


class SO2mConvolution(nnx.Module, PrecallInterface):
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
        num_experts: int = 0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.m_output_channels = m_output_channels
        self.num_experts = num_experts

        in_channels = (lmax - m + 1) * sphere_channels
        out_channels = 2 * (lmax - m + 1) * m_output_channels
        if num_experts > 0:
            self.fc = MoLE(
                num_experts, in_channels, out_channels, use_bias=False,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        else:
            self.fc = nnx.Linear(
                in_channels, out_channels, use_bias=False,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

    @PrecallInterface.context_handler
    def __call__(
        self,
        edge_feats_m: jax.Array,
        *,
        ctx: dict | None = None,
    ):
        num_edges = len(edge_feats_m)

        if self.num_experts > 0:
            edge_feats_m = self.fc(edge_feats_m, ctx=ctx)
        else:
            edge_feats_m = self.fc(edge_feats_m)
        edge_feats_r, edge_feats_i = jnp.split(edge_feats_m, 2, axis=2)
        edge_feats_m_r = edge_feats_r[:, 0] - edge_feats_i[:, 1]
        edge_feats_m_i = edge_feats_r[:, 1] + edge_feats_i[:, 0]

        return (
            edge_feats_m_r.reshape(num_edges, -1, self.m_output_channels),
            edge_feats_m_i.reshape(num_edges, -1, self.m_output_channels),
        )


class SO2Convolution(nnx.Module, PrecallInterface):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int): Number of spherical channels
        m_output_channels (int): Number of output channels used during the SO(2) conv
        mapping_coeffs (MappingCoefficients): Coefficients to convert l and m indices
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
        num_experts: int = 0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.m_output_channels = m_output_channels
        self.lmax = mapping_coeffs.lmax
        self.mmax = mapping_coeffs.mmax
        self.m_size = mapping_coeffs.m_size
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels
        self.num_experts = num_experts

        num_channels_m0 = (self.lmax + 1) * sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = (self.lmax + 1) * m_output_channels

        if extra_m0_output_channels is not None:
            m0_output_channels = (
                m0_output_channels + extra_m0_output_channels
            )
        if num_experts > 0:
            self.fc_m0 = MoLE(
                num_experts, num_channels_m0, m0_output_channels,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        else:
            self.fc_m0 = nnx.Linear(
                num_channels_m0, m0_output_channels,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        num_channels_rad = self.fc_m0.in_features # for radial function

        # SO(2) convolution for non-zero m
        so2_m_conv = []
        for m in range(1, self.mmax + 1):
            so2_m_conv.append(
                SO2mConvolution(
                    m,
                    sphere_channels,
                    m_output_channels,
                    self.lmax,
                    num_experts,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
            num_channels_rad = (
                num_channels_rad + so2_m_conv[-1].fc.in_features
            )
        self.so2_m_conv = nnx.List(so2_m_conv)

        # Embedding function of distance
        if not self.internal_weights:
            edge_channels_list = copy.deepcopy(edge_channels_list)
            if edge_channels_list is None:
                raise ValueError(
                    "If `internal_weights` is False, `edge_channels_list` must be provided."
                )
            edge_channels_list.append(int(num_channels_rad))
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

    @PrecallInterface.context_handler
    def __call__(
        self,
        edge_feats: jax.Array, # in m primary order
        edge_embeds: jax.Array,
        *,
        ctx: dict | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        num_edges = len(edge_embeds)

        # radial function
        if not self.internal_weights:
            edge_embeds = self.rad_func(edge_embeds)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        edge_feats_0 = edge_feats[:, :self.m_size[0]]
        edge_feats_0 = edge_feats_0.reshape(num_edges, -1)
        if not self.internal_weights:
            edge_embeds_0 = edge_embeds[:, :self.fc_m0.in_features]
            edge_feats_0 = edge_feats_0 * edge_embeds_0
        if self.num_experts > 0:
            edge_feats_0 = self.fc_m0(edge_feats_0, ctx=ctx)
        else:
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
            if not self.internal_weights:
                edge_embeds_m = edge_embeds[
                    :,
                    offset_rad : self.so2_m_conv[m - 1].fc.in_features + offset_rad
                ]
                edge_embeds_m = edge_embeds_m.reshape(
                    num_edges, 1, self.so2_m_conv[m - 1].fc.in_features
                )
                edge_feats_m = edge_feats_m * edge_embeds_m
            edge_feats_m = self.so2_m_conv[m - 1](edge_feats_m, ctx=ctx)
            # x[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = edge_feats_m
            edge_feats_out.extend(edge_feats_m)
            offset = offset + 2 * self.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

        edge_feats = jnp.concat(edge_feats_out, axis=1)

        if self.extra_m0_output_channels is not None:
            return edge_feats, edge_feats_0_extra
        return edge_feats
