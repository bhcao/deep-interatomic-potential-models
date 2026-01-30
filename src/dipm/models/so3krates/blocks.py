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
from flax.typing import Dtype
from flax.nnx.nn import initializers, dtypes
import e3nn_jax as e3nn

from dipm.layers import get_activation_fn


class MLP(nnx.Module):
    """Multi-layer perceptrons."""

    def __init__(
        self,
        features: list[int],
        activation: str,
        use_bias: bool = True,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.activation = activation
        self.use_bias = use_bias
        self.dtype = dtype

        layers = []
        in_features = features[0]
        for out_features in features[1:]:
            layer = nnx.Linear(
                in_features, out_features, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            layers.append(layer)
            in_features = out_features

        self.layers = nnx.List(layers)
        self.activation_fn = get_activation_fn(activation)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_fn(x)
        return x


class ResidualMLP(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        num_blocks: int = 3,
        activation: str = 'silu',
        # In original So3krates, this is set to False. But using bias is better.
        use_bias: bool = True,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.activation = activation
        self.use_bias = use_bias
        self.dtype = dtype

        self.layers = nnx.List([
            nnx.Linear(
                num_channels, num_channels, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            for _ in range(num_blocks)
        ])
        self.activation_fn = get_activation_fn(activation)

    def __call__(self, inputs: jax.Array):
        x = inputs
        for layer in self.layers:
            x = get_activation_fn(self.activation)(x)
            x = layer(x)
        x = x + inputs
        # In original So3krates, there exists a non-residual Linear. But it would
        # be slightly better to include it in the residual.
        # x = get_activation_fn(self.activation)(x)
        # x = nn.Dense(feat, use_bias=self.use_bias)(x)
        return x


class InteractionBlock(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_channels = num_channels
        self.dtype = dtype

        self.linear = nnx.Linear(
            num_channels, num_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray,
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        num_features = node_feats.shape[-1]

        # Tensor product using CG coefficents has been removed for simplicity.
        chi_scalar = e3nn.norm(chi, squared=True, per_irrep=True).array

        feats = jnp.concatenate([node_feats, chi_scalar], axis=-1)
        feats = self.linear(feats)

        # node_feats: [n_nodes, num_features], chi_coeffs: [n_nodes, n_heads]
        node_feats, chi_coeffs = jnp.split(feats, [num_features], axis=-1)

        return node_feats, chi_coeffs * chi


class FeatureBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        rad_features: list[int],
        sph_features: list[int],
        activation: str,
        avg_num_neighbors: float,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.rad_features = rad_features
        self.sph_features = sph_features
        self.activation = activation
        self.avg_num_neighbors = avg_num_neighbors
        self.dtype = dtype

        self.attn_map = FilterScaledAttentionMap(
            num_heads=num_heads,
            rad_features=rad_features,
            sph_features=sph_features,
            activation=activation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear = nnx.Linear(
            rad_features[-1], rad_features[-1], use_bias=False,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> jax.Array:
        alpha = self.attn_map(
            node_feats, edge_feats, chi_scalar, senders, receivers
        ) * cutoffs[:, None] # [n_edges, n_heads]

        head_dim = node_feats.shape[-1] // self.num_heads
        v_j = self.linear(node_feats)[senders].reshape(-1, self.num_heads, head_dim)

        node_feats = jax.ops.segment_sum(
            alpha[..., None] * v_j, receivers, num_segments=node_feats.shape[0]
        ) / self.avg_num_neighbors
        node_feats = node_feats.reshape(-1, head_dim * self.num_heads)
        return node_feats


class GeometricBlock(nnx.Module):
    def __init__(
        self,
        chi_irreps: e3nn.Irreps,
        rad_features: list[int],
        sph_features: list[int],
        activation: str,
        avg_num_neighbors: float,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.rad_features = rad_features
        self.sph_features = sph_features
        self.activation = activation
        self.avg_num_neighbors = avg_num_neighbors
        self.dtype = dtype

        self.attn_map = FilterScaledAttentionMap(
            num_heads=chi_irreps.num_irreps,
            rad_features=rad_features,
            sph_features=sph_features,
            activation=activation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        edge_sh: e3nn.IrrepsArray,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> e3nn.IrrepsArray:
        alpha = self.attn_map(
            node_feats, edge_feats, chi_scalar, senders, receivers
        ) * cutoffs[:, None]

        # e3nn supports directly multiply IrrepsArray with scalars.
        chi = e3nn.scatter_sum(alpha * edge_sh, dst=receivers, output_size=node_feats.shape[0])
        return chi / self.avg_num_neighbors


class FilterScaledAttentionMap(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        rad_features: list[int],
        sph_features: list[int],
        activation: str,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.rad_features = rad_features
        self.sph_features = sph_features
        self.activation = activation
        self.dtype = dtype

        assert rad_features[-1] == sph_features[-1]

        self.rad_mlp = MLP(
            rad_features, activation,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.sph_mlp = MLP(
            sph_features, activation,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        self.to_k = nnx.Linear(
            rad_features[-1], rad_features[-1], use_bias=False,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.to_q = nnx.Linear(
            rad_features[-1], rad_features[-1], use_bias=False,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> jax.Array:
        head_dim = node_feats.shape[-1] // self.num_heads

        # Radial spherical filter
        w_ij = self.rad_mlp(edge_feats)
        w_ij += self.sph_mlp(chi_scalar)
        w_ij = w_ij.reshape(-1, self.num_heads, head_dim)

        # Geometric attention coefficients
        q_i = self.to_q(node_feats).reshape(-1, self.num_heads, head_dim)[receivers]
        k_j = self.to_k(node_feats).reshape(-1, self.num_heads, head_dim)[senders]

        return (q_i * w_ij * k_j).sum(axis=-1) / jnp.sqrt(head_dim) # [n_edges, n_heads]


class ZBLRepulsion(nnx.Module):
    """Ziegler-Biersack-Littmark repulsion."""

    def __init__(
        self,
        index_to_z: list[int],
        a0: float = 0.5291772105638411,
        ke: float = 14.399645351950548,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.index_to_z = jnp.array(index_to_z)
        self.ke = ke

        def softplus_inverse(x):
            return x + jnp.log(-jnp.expm1(-x))

        # We vectorize a/c for simplicity.
        a_init = softplus_inverse(jnp.array([3.20000, 0.94230, 0.40280, 0.20160]))
        c_init = softplus_inverse(jnp.array([0.18180, 0.50990, 0.28020, 0.02817]))

        self.a = nnx.Param(initializers.constant(a_init)(rngs.params(), (4,), param_dtype))
        self.c = nnx.Param(initializers.constant(c_init)(rngs.params(), (4,), param_dtype))
        self.p = nnx.Param(
            initializers.constant(softplus_inverse(0.23))(rngs.params(), (1,), param_dtype)
        )
        self.d = nnx.Param(
            initializers.constant(
                softplus_inverse(1 / (0.8854 * a0))
            )(rngs.params(), (1,), param_dtype)
        )
        self.dtype = dtype

    def __call__(
        self,
        node_species: jax.Array,
        distances: jax.Array,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        distances, cutoffs, index_to_z, a, c, p, d = dtypes.promote_dtype(
            (distances, cutoffs, self.index_to_z, self.a, self.c, self.p, self.d), dtype=self.dtype
        )

        z = index_to_z[node_species]
        z_i = z[receivers]
        z_j = z[senders]

        x = self.ke * cutoffs * z_i * z_j / (distances + 1e-8)

        p = nnx.softplus(p)
        rzd = distances * (jnp.power(z_i, p) + jnp.power(z_j, p)) * nnx.softplus(d)

        # ZBL screening function, shape: [n_edges]
        c = nnx.softplus(c)
        y = jnp.sum(c / jnp.sum(c) * jnp.exp(-nnx.softplus(a) * rzd[:, None]), axis=-1)

        scaled_d = distances / 1.5
        sigma_d = jnp.exp(-1. / (jnp.where(scaled_d > 1e-8, scaled_d, 1e-8)))
        sigma_1_d = jnp.exp(-1. / (jnp.where(1 - scaled_d > 1e-8, 1 - scaled_d, 1e-8)))
        w = sigma_1_d / (sigma_1_d + sigma_d)

        energy_rep = w * x * y / 2
        return jax.ops.segment_sum(energy_rep, receivers, num_segments=node_species.shape[0])
