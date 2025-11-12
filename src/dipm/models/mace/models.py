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

from collections.abc import Callable

import e3nn_jax as e3nn
from flax import nnx
from flax.typing import Dtype
import jax
import jax.numpy as jnp

from dipm.data.dataset_info import DatasetInfo
from dipm.models.atomic_energies import get_atomic_energies
from dipm.layers import (
    Linear,
    FullyConnectedTensorProduct,
    RadialEmbeddingLayer,
    RadialEnvelope,
    get_radial_basis_fn,
    get_radial_envelope_cls,
)
from dipm.models.mace.blocks import (
    LinearNodeEmbeddingLayer,
    EquivariantProductBasisBlock,
    InteractionBlock,
    NonLinearReadoutBlock,
)
from dipm.models.mace.config import MaceConfig
from dipm.models.force_model import ForceModel
from dipm.utils.safe_norm import safe_norm
from dipm.typing import get_dtype


class Mace(ForceModel):
    """The MACE model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner,
          and Gábor Csányi. Mace: Higher order equivariant message passing
          neural networks for fast and accurate force fields, 2023.
          URL: https://arxiv.org/abs/2206.07697.

    Attributes:
        config: Hyperparameters / configuration for the MACE model, see
                :class:`~dipm.models.mace.config.MaceConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = MaceConfig
    config: MaceConfig

    def __init__(
        self,
        config: dict | MaceConfig,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
        rngs: nnx.Rngs
    ):
        super().__init__(config, dataset_info, dtype=dtype)
        dtype = self.dtype
        param_dtype = get_dtype(self.config.param_dtype)

        e3nn.config("path_normalization", "path")
        e3nn.config("gradient_normalization", "path")

        r_max = self.dataset_info.cutoff_distance_angstrom

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

        avg_r_min = self.config.avg_r_min
        if avg_r_min is None:
            avg_r_min = self.dataset_info.avg_r_min_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        radial_envelope_cls = get_radial_envelope_cls(self.config.radial_envelope)
        if self.config.radial_envelope == RadialEnvelope.POLYNOMIAL:
            radial_envelope_fun = radial_envelope_cls(r_max, exponent=self.config.polymomial_degree)
        else:
            radial_envelope_fun = radial_envelope_cls(r_max)

        node_symmetry = self.config.node_symmetry
        if node_symmetry is None:
            node_symmetry = self.config.l_max
        elif node_symmetry > self.config.l_max:
            raise ValueError("Message symmetry must be lower or equal to 'l_max'")

        readout_mlp_irreps, output_irreps = self.config.readout_irreps

        mace_block_kwargs = dict(
            output_irreps=output_irreps,
            r_max=r_max,
            num_channels=self.config.num_channels,
            avg_num_neighbors=avg_num_neighbors,
            num_interactions=self.config.num_layers,
            avg_r_min=avg_r_min,
            num_species=num_species,
            num_bessel=self.config.num_bessel,
            radial_basis=get_radial_basis_fn("bessel"),
            radial_envelope=radial_envelope_fun,
            symmetric_tensor_product_basis=self.config.symmetric_tensor_product_basis,
            off_diagonal=False,
            l_max=self.config.l_max,
            node_symmetry=node_symmetry,
            include_pseudotensors=self.config.include_pseudotensors,
            num_readout_heads=self.config.num_readout_heads,
            readout_mlp_irreps=readout_mlp_irreps,
            correlation=self.config.correlation,
            activation=self.config.activation,
            gate_nodes=self.config.gate_nodes,
            species_embedding_dim=self.config.species_embedding_dim,
        )

        self.mace_block = MaceBlock(
            **mace_block_kwargs,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.atomic_energies = nnx.Cache(get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species, dtype=dtype
        ))

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        _n_node: jax.Array, # Nel version of pyg.Data.batch, not used
        _rngs: nnx.Rngs | None = None, # Rngs for dropout, None for eval, not used
    ) -> jax.Array:

        # [n_nodes, num_interactions, num_heads, 0e]
        contributions = self.mace_block(edge_vectors, node_species, senders, receivers)
        # [n_nodes, num_interactions, num_heads]
        contributions = contributions.array[:, :, :, 0]

        sum_over_heads = jnp.sum(contributions, axis=2)  # [n_nodes, num_interactions]
        node_energies = jnp.sum(sum_over_heads, axis=1)  # [n_nodes, ]

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies
        node_energies += self.atomic_energies.value[node_species]  # [n_nodes, ]

        return node_energies


class MaceBlock(nnx.Module):
    def __init__(
        self,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        r_max: float,
        num_interactions: int,  # Number of interactions (layers), default 2
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        radial_basis: Callable[[jax.Array], jax.Array],
        radial_envelope: Callable[[jax.Array], jax.Array],
        num_channels: int,
        num_bessel: int = 8,
        avg_r_min: float = None,
        l_max: int = 3,  # Max spherical harmonic degree, default 3
        node_symmetry: int = 1,  # Max degree of node features after cluster expansion
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        activation: str = "silu",  # activation function
        soft_normalization: float | None = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        include_pseudotensors: bool = False,
        num_readout_heads: int = 1,
        residual_connection_first_layer: bool = False,
        gate_nodes: bool = False,
        species_embedding_dim: int | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        output_irreps = e3nn.Irreps(output_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        # Embeddings
        self.node_embed = LinearNodeEmbeddingLayer(
            num_species,
            num_channels * e3nn.Irreps("0e"),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.radial_embed = RadialEmbeddingLayer(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
            num_bessel=num_bessel,
        )
        self.species_embed = None
        if species_embedding_dim is not None:
            self.species_embed = nnx.Embed(
                num_species, species_embedding_dim,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )

        # Target of EquivariantProductBasisBlock and skip-connections
        node_irreps = e3nn.Irreps.spherical_harmonics(node_symmetry)

        # Target of InteractionBlock = source of EquivariantProductBasisBlock
        if not include_pseudotensors:
            interaction_irreps = e3nn.Irreps.spherical_harmonics(l_max)
        else:
            interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(l_max))

        in_irreps = num_channels * e3nn.Irreps("0e")
        self.layers = []
        for i in range(num_interactions):
            selector_tp = (i == 0) and not residual_connection_first_layer
            last_layer = i == num_interactions - 1

            layer = MaceLayer(
                in_irreps=in_irreps,
                selector_tp=selector_tp,
                last_layer=last_layer,
                num_channels=num_channels,
                node_irreps=node_irreps,
                interaction_irreps=interaction_irreps,
                l_max=l_max,
                avg_num_neighbors=avg_num_neighbors,
                activation=activation,
                num_species=num_species,
                correlation=correlation,
                output_irreps=output_irreps,
                readout_mlp_irreps=readout_mlp_irreps,
                symmetric_tensor_product_basis=symmetric_tensor_product_basis,
                off_diagonal=off_diagonal,
                soft_normalization=soft_normalization,
                num_readout_heads=num_readout_heads,
                radial_embedding_dim=num_bessel,
                species_embedding_dim=species_embedding_dim,
                gate_nodes=gate_nodes,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            in_irreps = layer.out_irreps
            self.layers.append(layer)

    def __call__(
        self,
        edge_vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        node_mask: jax.Array | None = None,  # [n_nodes] only used for profiling
    ) -> e3nn.IrrepsArray:
        assert edge_vectors.ndim == 2 and edge_vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert edge_vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embed(node_species)  # [n_nodes, feature * irreps]

        if not (hasattr(edge_vectors, "irreps") and hasattr(edge_vectors, "array")):
            edge_vectors = e3nn.IrrepsArray("1o", edge_vectors)

        radial_embeddings = self.radial_embed(safe_norm(edge_vectors.array, axis=-1))

        # Node and edge species features
        if self.species_embed is not None:
            node_species_feat = self.species_embed(node_species)
            vmap_multiply = jax.vmap(jnp.multiply)

            edge_species_feat = vmap_multiply(
                node_species_feat[senders], node_species_feat[receivers]
            )

            edge_species_feat = jnp.concat(
                [
                    node_species_feat[senders],
                    node_species_feat[receivers],
                    edge_species_feat,
                ],
                axis=-1,
            )
        else:
            edge_species_feat = None

        # Interactions
        outputs = []
        for layer in self.layers:
            node_outputs, node_feats = layer(
                edge_vectors,
                node_feats,
                node_species,
                radial_embeddings,
                senders,
                receivers,
                node_mask,
                edge_species_feat,
            )
            outputs += [node_outputs]  # list of [n_nodes, num_heads, output_irreps]

        return e3nn.stack(
            outputs, axis=1
        )  # [n_nodes, num_interactions, num_heads, output_irreps]


class MaceLayer(nnx.Module):
    def __init__(
        self,
        selector_tp: bool,
        last_layer: bool,
        num_channels: int,
        in_irreps: e3nn.Irreps,
        node_irreps: e3nn.Irreps,
        interaction_irreps: e3nn.Irreps,
        activation: str,
        num_species: int,
        # InteractionBlock:
        l_max: int,
        avg_num_neighbors: float,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        soft_normalization: float | None,
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
        num_readout_heads: int,
        radial_embedding_dim: int,
        species_embedding_dim: int | None = None,
        gate_nodes: bool = False,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.selector_tp = selector_tp
        self.last_layer = last_layer
        self.num_species = num_species
        self.soft_normalization = soft_normalization

        in_irreps = e3nn.Irreps(in_irreps)
        node_irreps = e3nn.Irreps(node_irreps)
        interaction_irreps = e3nn.Irreps(interaction_irreps)
        output_irreps = e3nn.Irreps(output_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        # Setting output_irreps
        if last_layer:
            out_irreps = num_channels * e3nn.Irreps("0e")
        else:
            out_irreps = num_channels * node_irreps.regroup()

        if not selector_tp:
            self.residual_connection = FullyConnectedTensorProduct(
                in_irreps,
                num_species * e3nn.Irreps("0e"),
                out_irreps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.interaction = InteractionBlock(
            in_irreps,
            num_channels * interaction_irreps,
            avg_num_neighbors=avg_num_neighbors,
            l_max=l_max,
            activation=activation,
            radial_embedding_dim=radial_embedding_dim,
            species_embedding_dim=species_embedding_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # selector tensor product (first layer only)
        if selector_tp:
            self.selector_tp_layer = FullyConnectedTensorProduct(
                self.interaction.out_irreps,
                num_species * e3nn.Irreps("0e"),
                self.interaction.out_irreps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        # Exponentiate node features, keep degrees < node_symmetry only
        self.equivariant_product = EquivariantProductBasisBlock(
            self.interaction.out_irreps,
            target_irreps=out_irreps,
            correlation=correlation,
            num_species=num_species,
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
            off_diagonal=off_diagonal,
            gate_nodes=gate_nodes,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.readout_layers = []
        if not last_layer:
            for _head_idx in range(num_readout_heads):
                self.readout_layers.append(Linear(
                    out_irreps,
                    output_irreps,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ))  # [n_nodes, output_irreps]
        else:  # Non-linear readout for last layer
            for _head_idx in range(num_readout_heads):
                self.readout_layers.append(NonLinearReadoutBlock(
                    out_irreps,
                    readout_mlp_irreps,
                    output_irreps,
                    activation=activation,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ))  # [n_nodes, output_irreps]

        # output irreps of node_feats
        self.out_irreps = out_irreps

    def __call__(
        self,
        edge_vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        radial_embeddings: jax.Array,  # [n_edges, radial_embedding_dim]
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        node_mask: jax.Array | None = None,  # [n_nodes] only used for profiling
        edge_species_feat: jax.Array | None = None,  # [n_edges, species_embedding_dim * 3]
    ):
        identity = jnp.eye(self.num_species)
        node_attr = e3nn.IrrepsArray(self.num_species * e3nn.Irreps("0e"), identity[node_species])

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # residual connection:
        residual_connection = None

        if not self.selector_tp:
            residual_connection = self.residual_connection(node_feats, node_attr)

        # Interaction block
        node_feats = self.interaction(
            edge_vectors=edge_vectors,
            node_feats=node_feats,
            radial_embeddings=radial_embeddings,
            receivers=receivers,
            senders=senders,
            edge_species_feat=edge_species_feat,
        )

        # selector tensor product (first layer only)
        if self.selector_tp:
            node_feats = self.selector_tp_layer(node_feats, node_attr)

        # Exponentiate node features, keep degrees < node_symmetry only
        node_feats = self.equivariant_product(node_feats, node_species)

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(
                node_feats, [phi] * len(node_feats.irreps)
            )
        if residual_connection is not None:
            node_feats = (
                node_feats + residual_connection
            )  # [n_nodes, feature * hidden_irreps]

        # Multi-head readout
        node_outputs = []

        for readout_layer in self.readout_layers:
            node_outputs += [readout_layer(node_feats)]  # [n_nodes, output_irreps]

        node_outputs = e3nn.stack(
            node_outputs, axis=1
        )  # [n_nodes, num_heads, output_irreps]

        return node_outputs, node_feats
