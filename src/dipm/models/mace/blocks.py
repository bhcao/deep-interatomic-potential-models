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
from flax.nnx.nn import initializers
from flax.typing import Dtype, Initializer
import jax
import jax.numpy as jnp

from dipm.layers import (
    Linear,
    MultiLayerPerceptron,
    get_activation_fn,
)
from dipm.models.mace.symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingLayer(nnx.Module):
    def __init__(
        self,
        num_species: int,
        irreps_out: e3nn.Irreps,
        *,
        param_dtype: Dtype = jnp.float32,
        embeddings_init: Initializer = initializers.normal(stddev=1.0),
        rngs: nnx.Rngs,
    ):
        self.num_species = num_species
        self.irreps_out = irreps_out.filter("0e").regroup()

        key = rngs.params()
        self.embeddings = nnx.Param(
            embeddings_init(
                key, (num_species, self.irreps_out.dim), param_dtype
            )
        )

    def __call__(self, node_specie: jax.Array) -> e3nn.IrrepsArray:
        irreps_out = self.irreps_out

        w = (1 / jnp.sqrt(self.num_species)) * self.embeddings.value
        return e3nn.IrrepsArray(irreps_out, w[node_specie])


class NonLinearReadoutBlock(nnx.Module):
    def __init__(
        self,
        input_irreps: e3nn.Irreps | str,
        hidden_irreps: e3nn.Irreps | str,
        output_irreps: e3nn.Irreps | str,
        activation: str | None = None,
        gate_activation: str | None = None,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.activation = None if activation is None else get_activation_fn(activation)
        self.gate_activation = None if gate_activation is None else get_activation_fn(gate_activation)

        input_irreps = e3nn.Irreps(input_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        output_irreps = e3nn.Irreps(output_irreps)

        num_vectors = hidden_irreps.filter(
            drop=["0e", "0o"]
        ).num_irreps

        intermediate_irreps = (hidden_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify()

        self.linear_in = Linear(
            input_irreps,
            intermediate_irreps,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        intermediate_irreps = e3nn.gate(intermediate_irreps)

        self.linear_out = Linear(
            intermediate_irreps,
            output_irreps,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # x = [n_nodes, irreps]
        # Multiplicity of (l > 0) irreps
        x = self.linear_in(x)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate_activation)
        return self.linear_out(x)  # [n_nodes, output_irreps]


class EquivariantProductBasisBlock(nnx.Module):
    def __init__(
        self,
        node_irreps: e3nn.Irreps | str,
        target_irreps: e3nn.Irreps | str,
        correlation: int,
        num_species: int,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        gate_nodes: bool = False,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.gate_nodes = gate_nodes

        node_irreps = e3nn.Irreps(node_irreps)
        target_irreps = e3nn.Irreps(target_irreps)

        num_features = node_irreps.mul_gcd
        self.contraction = SymmetricContraction(
            num_features=num_features,
            node_irreps=node_irreps // num_features,
            correlation=correlation,
            keep_irrep_out={ir for _, ir in target_irreps},
            num_species=num_species,
            gradient_normalization="element",
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
            off_diagonal=off_diagonal,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if gate_nodes:
            scaler_num = target_irreps.filter("0e").num_irreps
            key = rngs.params()
            self.gate_kernel = nnx.Param(
                initializers.normal(stddev=1 / jnp.sqrt(scaler_num))(
                    key, (num_species, scaler_num, target_irreps.num_irreps), param_dtype
                )
            )
            key = rngs.params()
            self.gate_bias = nnx.Param(
                initializers.normal()(
                    key, (num_species, target_irreps.num_irreps), param_dtype
                )
            )

        self.linear_out = Linear(
            target_irreps,
            target_irreps,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def node_gating(
        self, node_feats: e3nn.IrrepsArray, node_species: jax.Array
    ) -> e3nn.IrrepsArray:
        node_scalars = node_feats.filter("0e").array
        w = self.gate_kernel.value[node_species]
        b = self.gate_bias.value[node_species]
        node_feats = node_feats * (jax.vmap(jnp.matmul)(node_scalars, w) + b)
        return node_feats

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature * irreps]
        node_species: jax.Array,  # [n_nodes, ] int
    ) -> e3nn.IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_zero_chunks()
        node_feats = self.contraction(node_feats, node_species)
        node_feats = node_feats.axis_to_mul()

        if self.gate_nodes:
            node_feats = self.node_gating(node_feats, node_species)

        return self.linear_out(node_feats)


class MessagePassingConvolution(nnx.Module):
    def __init__(
        self,
        in_irreps: e3nn.Irreps | str,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps | str,
        l_max: int,
        activation: str,
        radial_embedding_dim: int,
        species_embedding_dim: int | None = None,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.l_max = l_max

        in_irreps = e3nn.Irreps(in_irreps)
        out_irreps = in_irreps.filter(self.target_irreps) + e3nn.tensor_product(
            in_irreps,
            # Spherical harmonics from 1 to l_max
            e3nn.Irreps([(1, (l, (-1)**l)) for l in range(1, l_max + 1)]),
            filter_ir_out=self.target_irreps,
        )
        out_irreps = out_irreps.regroup()

        self.mix = MultiLayerPerceptron(
            [radial_embedding_dim] + 3 * [64] + [out_irreps.num_irreps],
            activation,
            gradient_normalization=1.0,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.mix_species = None
        if species_embedding_dim is not None:
            self.mix_species = MultiLayerPerceptron(
                [3*species_embedding_dim] + 3 * [64] + [out_irreps.num_irreps],
                activation,
                gradient_normalization=1.0,
                use_bias=True,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.out_irreps = out_irreps

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jax.Array,  # [n_edges, radial_embedding_dim]
        senders: jax.Array,  # [n_edges, ]
        receivers: jax.Array,  # [n_edges, ]
        edge_species_feat: jax.Array | None = None,  # [n_edges, species_embedding_dim * 3]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2
        if self.mix_species is not None:
            assert edge_species_feat is not None

        messages = node_feats[senders]
        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(range(1, self.l_max + 1), -vectors, True),
                    filter_ir_out=self.target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        mix = self.mix(radial_embedding)  # [n_edges, num_irreps]

        if self.mix_species is not None:
            mix_species = self.mix_species(edge_species_feat)  # [n_edges, num_irreps]
            mix = jax.vmap(jnp.multiply)(mix.array, mix_species)

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / self.avg_num_neighbors


class InteractionBlock(nnx.Module):
    def __init__(
        self,
        node_irreps: e3nn.Irreps | str,
        target_irreps: e3nn.Irreps | str,
        avg_num_neighbors: float,
        l_max: int,
        activation: str,
        radial_embedding_dim: int,
        species_embedding_dim: int | None = None,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.species_embedding_dim = species_embedding_dim

        node_irreps = e3nn.Irreps(node_irreps)
        target_irreps = e3nn.Irreps(target_irreps)

        self.linear_up = Linear(node_irreps, node_irreps, param_dtype=param_dtype, rngs=rngs)
        self.conv = MessagePassingConvolution(
            node_irreps,
            avg_num_neighbors,
            target_irreps,
            l_max,
            activation,
            radial_embedding_dim=radial_embedding_dim,
            species_embedding_dim=species_embedding_dim,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear_down = Linear(self.conv.out_irreps, target_irreps, param_dtype=param_dtype, rngs=rngs)

    def __call__(
        self,
        edge_vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embeddings: jax.Array,  # [n_edges, radial_embedding_dim]
        senders: jax.Array,  # [n_edges, ]
        receivers: jax.Array,  # [n_edges, ]
        edge_species_feat: jax.Array | None = None,  # [n_edges, species_embedding_dim * 3]
    ) -> tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        assert node_feats.ndim == 2
        assert edge_vectors.ndim == 2
        assert radial_embeddings.ndim == 2
        if self.species_embedding_dim is not None:
            assert edge_species_feat is not None

        node_feats = self.linear_up(node_feats)

        node_feats = self.conv(
            edge_vectors,
            node_feats,
            radial_embeddings,
            senders,
            receivers,
            edge_species_feat,
        )
        node_feats = self.linear_down(node_feats)

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]
