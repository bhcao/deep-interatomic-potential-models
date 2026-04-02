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

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers
import e3nn_jax as e3nn

from dipm.layers import dtypes, get_activation_fn


def _check_irreps_aligned(irreps: e3nn.Irreps) -> tuple[int, e3nn.Irreps]:
    mul_in = set(m for m, _ in irreps.regroup())
    assert len(mul_in) == 1, "Input irreps must have the same multiplicity."

    mul_in, = mul_in
    irreps_out = irreps.regroup() // mul_in
    assert e3nn.Irreps([i for _ in range(mul_in) for i in irreps_out]) == irreps, (
        "Input irreps must be in order `1e+2e+...+1e+2e+...`"
    )

    return mul_in, irreps_out


class AlignedLinear(nnx.Module):
    r"""Aligned equivariant Linear. A fast version of e3nn.flax.Linear.

    - input irreps = $mul \times (l, p)$
    - output irreps = $mul_out \times (l, p)$
    """
    def __init__(
        self,
        in_irreps: e3nn.Irreps,
        mul_out: int,
        split: int = 1,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        mul_in, irreps = _check_irreps_aligned(in_irreps)
        assert split > 0, "split must be positive."
        assert mul_out % split == 0, "mul_out must be divisible by split."

        self.irreps = irreps
        self.in_irreps = in_irreps
        self.mul_in = mul_in
        self.split = split
        self.out_irreps = e3nn.Irreps(
            [i for _ in range(mul_out // self.split) for i in irreps]
        )
        self.dtype = dtype

        gradient_normalization = e3nn.config("gradient_normalization")
        if gradient_normalization == "element":
            self.alpha = 1.0
            initializer = initializers.lecun_normal()
        elif gradient_normalization == "path":
            self.alpha = np.sqrt(1.0 / mul_in)
            initializer = initializers.normal(stddev=1.0)
        else:
            raise ValueError(f"Unknown gradient_normalization: {gradient_normalization}")

        self.kernel = nnx.Param(
            initializer(rngs.params(), (len(irreps), mul_in, mul_out), param_dtype)
        )

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        assert x.irreps == self.in_irreps, "Input irreps must match the module's input irreps."

        x, kernel = dtypes.promote_dtype((x, self.kernel.value), dtype=self.dtype)

        repeats = np.array([ir.dim for _, ir in self.irreps])
        kernel = jnp.repeat(kernel, repeats, axis=0)

        x_arr = x.array.reshape(*x.shape[:-1], self.mul_in, -1)

        y = self.alpha * jnp.einsum(
            "...im, mio -> ...om", x_arr, kernel
        ).reshape(*x.shape[:-1], -1)

        if self.split == 1:
            return e3nn.IrrepsArray(self.out_irreps, y)

        return (
            e3nn.IrrepsArray(self.out_irreps, y_i)
            for y_i in jnp.split(y, self.split, axis=-1)
        )


def aligned_norm(x: e3nn.IrrepsArray) -> jax.Array:
    r"""e3nn.norm for aligned irreps."""
    mul_in, irreps = _check_irreps_aligned(x.irreps)

    x_norm = jnp.square(x.array).reshape(*x.shape[:-1], mul_in, -1)

    offset = 0
    outs = []
    for _, ir in irreps:
        outs.append(
            x_norm[..., offset:offset+ir.dim].sum(axis=-1)
        )
        offset += ir.dim
    x_scalar = jnp.stack(outs, axis=-1)

    return x_scalar.reshape(*x.shape[:-1], -1)


def aligned_dot(x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> jax.Array:
    r"""e3nn.dot for aligned irreps."""
    mul_in, irreps = _check_irreps_aligned(x.irreps)
    mul_in_y, irreps_y = _check_irreps_aligned(y.irreps)

    assert mul_in_y == mul_in, "Input irreps must have the same multiplicity."
    assert irreps == irreps_y, "Input irreps must be the same."

    x_norm = (x.array * y.array).reshape(*x.shape[:-1], mul_in, -1)

    offset = 0
    outs = []
    for _, ir in irreps:
        outs.append(
            x_norm[..., offset:offset+ir.dim].sum(axis=-1)
        )
        offset += ir.dim
    x_scalar = jnp.stack(outs, axis=-1)

    return x_scalar.reshape(*x.shape[:-1], -1)


def aligned_mul(x: e3nn.IrrepsArray, y: jax.Array) -> e3nn.IrrepsArray:
    r"""e3nn.IrrepsArray.__mul__ for aligned irreps."""
    mul_in, irreps = _check_irreps_aligned(x.irreps)
    assert x.irreps.num_irreps == y.shape[-1], "Input irreps and array must have the same shape."

    y_arr = y.reshape(*y.shape[:-1], mul_in, -1)
    y_arr = jnp.repeat(y_arr, np.array([ir.dim for _, ir in irreps]), axis=-1)

    x_arr = x.array.reshape(*x.shape[:-1], mul_in, -1)
    x_arr = (x_arr * y_arr).reshape(*x.shape[:-1], -1)
    return e3nn.IrrepsArray(x.irreps, x_arr)


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
        in_irreps: e3nn.Irreps,
        scalar_num_scale: int | None = None,
        num_linear: int | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_channels = num_channels
        self.scalar_num_scale = scalar_num_scale
        self.num_linear = num_linear
        self.dtype = dtype

        num_chi_scalars = in_irreps.num_irreps
        if scalar_num_scale is not None:
            self.scalar_linear = AlignedLinear(
                in_irreps, 2 * in_irreps.regroup().mul_gcd * self.scalar_num_scale, split=2,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            num_chi_scalars *= self.scalar_num_scale

        num_chi_coeffs = in_irreps.num_irreps
        if num_linear is not None:
            num_chi_coeffs *= self.num_linear + 1
            self.out_linears = nnx.List([
                AlignedLinear(
                    in_irreps, in_irreps.regroup().mul_gcd,
                    dtype=dtype, param_dtype=param_dtype, rngs=rngs,
                )
                for _ in range(num_linear)
            ])

        self.linear = nnx.Linear(
            num_channels + num_chi_scalars, num_channels + num_chi_coeffs,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray,
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        num_features = node_feats.shape[-1]

        if self.scalar_num_scale is not None:
            chi_left, chi_right = self.scalar_linear(chi)
            chi_scalar = aligned_dot(chi_left, chi_right)
        else:
            chi_scalar = aligned_norm(chi)

        feats = jnp.concatenate([node_feats, chi_scalar], axis=-1)
        feats = self.linear(feats)

        # node_feats: [n_nodes, num_features], chi_coeffs: [n_nodes, n_heads]
        node_feats, chi_coeffs = jnp.split(feats, [num_features], axis=-1)

        if self.num_linear is not None:
            chi_coeffs = jnp.split(chi_coeffs, self.num_linear + 1, axis=-1)
            for coeff, lin in zip(chi_coeffs[:-1], self.out_linears):
                chi = lin(aligned_mul(chi, coeff))
            return node_feats, aligned_mul(chi, chi_coeffs[-1])

        return node_feats, aligned_mul(chi, chi_coeffs)


class FeatureBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        rad_features: list[int],
        sph_features: list[int],
        activation: str,
        avg_num_neighbors: float,
        first_layer: bool = False,
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
            first_layer=first_layer,
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
        chi_scalar: jax.Array | None, # None for first layer
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
        first_layer: bool = False,
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
            first_layer=first_layer,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        edge_sh: e3nn.IrrepsArray,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array | None, # None for first layer
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> e3nn.IrrepsArray:
        alpha = self.attn_map(
            node_feats, edge_feats, chi_scalar, senders, receivers
        ) * cutoffs[:, None]

        # e3nn supports directly multiply IrrepsArray with scalars.
        chi = e3nn.scatter_sum(
            aligned_mul(edge_sh, alpha), dst=receivers, output_size=node_feats.shape[0]
        )
        return chi / self.avg_num_neighbors


class FilterScaledAttentionMap(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        rad_features: list[int],
        sph_features: list[int],
        activation: str,
        first_layer: bool = False,
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
        if not first_layer:
            self.sph_mlp = MLP(
                sph_features, activation,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
        else:
            self.sph_mlp = nnx.data(None)

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
        chi_scalar: jax.Array | None, # None for first layer
        senders: jax.Array,
        receivers: jax.Array
    ) -> jax.Array:
        head_dim = node_feats.shape[-1] // self.num_heads

        # Radial spherical filter
        w_ij = self.rad_mlp(edge_feats)
        if self.sph_mlp is not None:
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
