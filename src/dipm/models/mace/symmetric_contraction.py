# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
#
# Copyright 2025 InstaDeep Ltd
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

import e3nn_jax as e3nn
from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers
import jax
import jax.numpy as jnp
from jax import vmap

from dipm.layers.dtypes import promote_dtype

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


class Contraction(nnx.Module):
    def __init__(
        self,
        order: int,
        num_features: int,
        node_irreps: e3nn.Irreps,
        keep_irrep_out: set[e3nn.Irrep],
        num_species: int,
        gradient_normalization: float,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.order = order
        self.gradient_normalization = gradient_normalization
        self.off_diagonal = off_diagonal

        if symmetric_tensor_product_basis:
            reduced_basis = e3nn.reduced_symmetric_tensor_product_basis(
                node_irreps, order, keep_ir=keep_irrep_out
            )
        else:
            reduced_basis = e3nn.reduced_tensor_product_basis(
                [node_irreps] * order, keep_ir=keep_irrep_out
            )

        self.weights = []
        for mul, _ in reduced_basis.irreps:
            # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
            key = rngs.params()
            w = nnx.Param(initializers.normal(
                stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
            )(key, (num_species, mul, num_features), param_dtype))
            self.weights.append(w)

        self.reduced_basis = nnx.Cache(reduced_basis.astype(dtype))
        self.dtype = dtype

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        index: jax.Array,
        out: dict[e3nn.Irrep, jax.Array],
    ) -> dict[e3nn.Irrep, jax.Array]:
        weights = [w.value for w in self.weights]
        weights, node_feats = promote_dtype(
            (weights, node_feats), dtype=self.dtype
        )

        out_new = {}
        if self.off_diagonal:
            x_ = jnp.roll(node_feats.array, A025582[self.order - 1])
        else:
            x_ = node_feats.array

        reduced_basis = self.reduced_basis.value

        for (mul, ir_out), u, weight in zip(
            reduced_basis.irreps, reduced_basis.chunks, weights
        ):
            # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
            w = weight[index]  # [multiplicity, num_features]
            w = w * (mul**-0.5) ** self.gradient_normalization  # normalize weights
            if ir_out not in out:
                out_new[ir_out] = jnp.einsum(
                    "...jki,kc,cj->c...i", u, w, x_
                ) # [num_features, (irreps_x.dim)^(order-1), ir_out.dim]
            else:
                out[ir_out] += jnp.einsum(
                    "...ki,kc->c...i", u, w
                )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

        # ((w3 x + w2) x + w1) x
        #  \----------------/
        #         out (in the normal case)
        for ir_out, array in out.items():
            out_new[ir_out] = jnp.einsum(
                "c...ji,cj->c...i", array, x_
            )  # [num_features, (irreps_x.dim)^(order-1), ir_out.dim]

        return out_new


class SymmetricContraction(nnx.Module):
    def __init__(
        self,
        num_features: int,
        node_irreps: e3nn.Irreps,
        correlation: int,
        keep_irrep_out: set[e3nn.Irrep],
        num_species: int,
        gradient_normalization: str | float | None = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.keep_irrep_out = keep_irrep_out

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
            # possibly a string now
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        self.contractions = []
        for order in range(correlation, 0, -1):  # correlation, ..., 1
            self.contractions.append(Contraction(
                order,
                num_features,
                node_irreps,
                self._keep_irrep_out,
                num_species,
                gradient_normalization,
                symmetric_tensor_product_basis,
                off_diagonal,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ))

    @property
    def _keep_irrep_out(self) -> e3nn.Irreps:
        """Parse `keep_irrep_out` attribute, possibly a string."""
        out = e3nn.Irreps(self.keep_irrep_out)
        if not all(mul == 1 for mul, _ in out):
            raise ValueError("Expecting mul = 1 for `keep_irrep_out` filter")
        return out

    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jax.Array
    ) -> e3nn.IrrepsArray:
        """Power expansion of node_feats, mapped through index-wise weights.

        This module should return the equivalent of

            B = W[index] @ (A + (A ⊗ A) + ... + A**(⊗ ν))

        where `A = node_feats`, and `W` represents learnable weights acting
        specie-index-wise and momentum-wise on the equivariant powers of
        the node features.
        """

        def fn(features: e3nn.IrrepsArray, index: jax.Array):
            '''
            This operation is parallel on the feature dimension (but each feature has its own parameters)
            This operation is an efficient implementation of
            vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), 
            tensor_product(x, x, x), ...])))(w, x) up to x power self.correlation

            Args:
                features: [num_features, irreps_x.dim]
                index: int
            '''
            assert index.ndim == 0
            out = {}
            for contraction in self.contractions:  # correlation, ..., 1
                out = contraction(features, index, out)
            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.from_chunks(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (features.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(node_feats.shape[:-2], index.shape)
        node_feats = node_feats.broadcast_to(shape + node_feats.shape[-2:])
        index = jnp.broadcast_to(index, shape)
        fn_mapped = fn
        for _ in range(node_feats.ndim - 2):
            fn_mapped = vmap(fn_mapped)
        return fn_mapped(node_feats, index)
