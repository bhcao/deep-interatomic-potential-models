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

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Dtype

from dipm.data.dataset_info import DatasetInfo
from dipm.layers import (
    CosineCutoff,
    get_activation_fn,
    get_veclayernorm_fn,
    get_rbf_cls,
)
from dipm.models.force_model import ForceModel
from dipm.models.atomic_energies import get_atomic_energies
from dipm.models.liten.config import LiTENConfig
from dipm.utils.safe_norm import safe_norm


class LiTEN(ForceModel):
    """The LiTEN model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Qun Su, Kai Zhu, Qiaolin Gou, Jintu Zhang, Renling Hu, Yurong Li,
          Yongze Wang, Hui Zhang, Ziyi You, Linlong Jiang, Yu Kang, Jike Wang,
          Chang-Yu Hsieh and Tingjun Hou. A Scalable and Quantum-Accurate
          Foundation Model for Biomolecular Force Field via Linearly Tensorized
          Quadrangle Attention. arXiv, Jul 2025.
          URL: https://arxiv.org/abs/2507.00884.

    Attributes:
        config: Hyperparameters / configuration for the LiTEN model, see
                :class:`~dipm.models.liten.config.LiTENConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = LiTENConfig
    config: LiTENConfig

    def __init__(
        self,
        config: dict | LiTENConfig,
        dataset_info: DatasetInfo,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        super().__init__(config, dataset_info)

        r_max = self.dataset_info.cutoff_distance_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        liten_kwargs = dict(
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            rbf_type="expnorm",
            trainable_rbf=self.config.trainable_rbf,
            activation=self.config.activation,
            cutoff=r_max,
            num_species=num_species,
        )

        self.liten_block = LiTENBlock(**liten_kwargs, param_dtype=param_dtype, rngs=rngs)
        self.atomic_energies = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        _n_node: jax.Array, # Nel version of pyg.Data.batch, not used
        _rngs: nnx.Rngs | None = None, # Rngs for dropout, None for eval, not used
    ) -> jax.Array:
        # edge connect
        edge_distances = safe_norm(edge_vectors, axis=-1) # [n_edges]
        norm_edge_vectors = edge_vectors / (edge_distances[:, None] + 1e-8) # ViSNet style

        node_energies = self.liten_block(
            norm_edge_vectors, edge_distances, node_species, senders, receivers
        )

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        node_energies += self.atomic_energies[node_species]  # [n_nodes, ]

        return node_energies


class LiTENBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int = 8,
        num_layers: int = 6,
        num_channels: int = 256,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff: float = 5.0,
        num_species: int = 5,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        self.node_embedding = nnx.Embed(
            num_species, num_channels, param_dtype=param_dtype, rngs=rngs
        )
        self.radial_embedding = get_rbf_cls(rbf_type)(
            cutoff, num_rbf, trainable_rbf, param_dtype=param_dtype, rngs=rngs
        )
        self.edge_embedding = nnx.Linear(
            num_rbf, num_channels, param_dtype=param_dtype, rngs=rngs
        )

        self.liten_layers = [
            LiTENLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                activation=activation,
                cutoff=cutoff,
                vecnorm_type="max_min" if idx > 0 else "none",
                update_edge=idx < num_layers - 1,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for idx in range(num_layers)
        ]

        self.out_norm = nnx.LayerNorm(
            num_channels, epsilon=1e-05, param_dtype=param_dtype, rngs=rngs
        )
        self.readout_energy = nnx.Sequential(
            nnx.Linear(num_channels, num_channels // 2, param_dtype=param_dtype, rngs=rngs),
            get_activation_fn(activation),
            nnx.Linear(num_channels // 2, 1, param_dtype=param_dtype, rngs=rngs),
        )

    def __call__(
        self,
        norm_edge_vectors: jax.Array,  # [n_edges, 3]
        edge_distances: jax.Array, # [n_edges]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
    ) -> jax.Array:

        # Embedding Layers
        node_scalar = self.node_embedding(node_species) # [n_nodes, num_channels]
        edge_feats = self.radial_embedding(edge_distances)
        edge_feats = self.edge_embedding(edge_feats)

        # [n_nodes, 3, num_channels]
        node_vector = jnp.zeros((node_scalar.shape[0], 3, node_scalar.shape[1]), node_scalar.dtype)

        for layer in self.liten_layers:
            node_scalar, node_vector, edge_feats = layer(
                node_scalar=node_scalar,
                node_vector=node_vector,
                senders=senders,
                receivers=receivers,
                edge_distances=edge_distances,
                edge_feats=edge_feats,
                norm_edge_vectors=norm_edge_vectors
            )

        node_scalar = self.out_norm(node_scalar)
        node_energies = self.readout_energy(node_scalar).squeeze(-1)

        return node_energies


class LiTENLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        activation: str,
        cutoff: float,
        vecnorm_type: str,
        update_edge: bool = True,
        eps: float = 1e-8,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.head_dim = num_channels // num_heads
        self.update_edge = update_edge

        self.layernorm = nnx.LayerNorm(
            num_channels, param_dtype=param_dtype, rngs=rngs
        )
        self.vec_layernorm = get_veclayernorm_fn(vecnorm_type, eps)

        self.act = get_activation_fn(activation)

        key = rngs.params()
        self.alpha = nnx.Param(
            initializers.xavier_normal()(key, (1, num_heads, self.head_dim), param_dtype)
        )

        self.cutoff_fn = CosineCutoff(cutoff)

        self.vec_linear = nnx.Linear(
            num_channels,
            num_channels * 2,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.cross_linear = nnx.Linear(
            num_channels,
            num_channels,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.node_linear = nnx.Linear(
            num_channels,
            num_channels,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.edge_linear = nnx.Linear(
            num_channels,
            num_channels,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.part_linear1 = nnx.Linear(
            num_channels,
            num_channels * 2,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.part_linear2 = nnx.Linear(
            num_channels,
            num_channels * 3,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.f_linear = nnx.Linear(
            num_channels,
            num_channels,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def edge_update(
        self,
        node_vector: jax.Array, # [n_nodes, 3, num_channels]
        senders: jax.Array, # [n_edges]
        receivers: jax.Array, # [n_edges]
        norm_edge_vectors: jax.Array, # [n_edges, 3]
        edge_feats: jax.Array, # [n_edges, num_channels]
    ):
        node_vector = self.cross_linear(node_vector)

        norm_edge_vectors = norm_edge_vectors[:, :, None]

        vec_cross_i = jnp.cross(node_vector[senders], norm_edge_vectors, axis=1)
        vec_cross_j = jnp.cross(node_vector[receivers], norm_edge_vectors, axis=1)
        sum_phi = jnp.sum(vec_cross_i * vec_cross_j, axis=1)

        delta_edge_feats = self.act(self.f_linear(edge_feats)) * sum_phi

        return delta_edge_feats

    def message_fn(
        self,
        node_scalar: jax.Array, # [n_nodes, num_channels]
        node_vector: jax.Array, # [n_nodes, 3, num_channels]
        senders: jax.Array, # [n_edges]
        receivers: jax.Array, # [n_edges]
        edge_distances: jax.Array, # [n_edges]
        norm_edge_vectors: jax.Array, # [n_edges, 3]
        edge_feats: jax.Array, # [n_edges, num_channels]
    ):
        edge_feats = self.act(self.edge_linear(edge_feats)).reshape(-1, self.num_heads, self.head_dim)
        node_scalar = self.node_linear(node_scalar).reshape(-1, self.num_heads, self.head_dim)
        attn = node_scalar[receivers] + node_scalar[senders] + edge_feats
        attn = self.act(attn) * self.alpha
        attn = attn.sum(axis=-1) * self.cutoff_fn(edge_distances)[:, None]
        attn = attn[:, :, None]

        n_nodes = len(node_scalar)
        node_scalar = node_scalar[senders] * edge_feats
        node_scalar = (node_scalar * attn).reshape(-1, self.num_channels)

        node_sca = self.act(self.part_linear1(node_scalar))[:, None] # [n_edges, 1, 2*num_channels]
        node_sca1, node_sca2 = jnp.split(node_sca, 2, axis=2)
        node_vector = node_vector[senders] * node_sca1 + node_sca2 * norm_edge_vectors[:, :, None]

        node_scalar = jax.ops.segment_sum(node_scalar, receivers, num_segments=n_nodes)
        node_vector = jax.ops.segment_sum(node_vector, receivers, num_segments=n_nodes)

        return node_scalar, node_vector

    def node_update(
        self,
        node_scalar: jax.Array, # [n_nodes, num_channels]
        node_vector: jax.Array, # [n_nodes, 3, num_channels]
    ):
        node_vec1, node_vec2 = jnp.split(self.vec_linear(node_vector), 2, axis=-1)
        vec_tri = jnp.sum(node_vec1 * node_vec2, axis=1)

        norm_vec = jnp.sqrt(jnp.sum(node_vec2 ** 2, axis=-2) + 1e-8)
        vec_qua = norm_vec ** 3

        node_scalar = self.part_linear2(node_scalar)

        node_sca1, node_sca2, node_sca3 = jnp.split(node_scalar, 3, axis=1)

        delta_scalar = (vec_qua + vec_tri) * node_sca1 + node_sca2
        delta_vector = node_vec1 * node_sca3[:, None]

        return delta_scalar, delta_vector

    def __call__(
        self,
        node_scalar: jax.Array, # [n_nodes, num_channels]
        node_vector: jax.Array, # [n_nodes, 3, num_channels]
        senders: jax.Array, # [n_edges]
        receivers: jax.Array, # [n_edges]
        edge_distances: jax.Array, # [n_edges]
        edge_feats: jax.Array, # [n_edges, num_channels]
        norm_edge_vectors: jax.Array, # [n_edges, 3]
    ):
        scalar_out = self.layernorm(node_scalar)
        node_vector = self.vec_layernorm(node_vector)

        scalar_out, vector_out = self.message_fn(
            scalar_out, node_vector, senders, receivers, edge_distances, norm_edge_vectors, edge_feats
        )

        if self.update_edge:
            delta_edge_feats = self.edge_update(
                node_vector, senders, receivers, norm_edge_vectors, edge_feats
            )
            edge_feats = edge_feats + delta_edge_feats

        node_scalar = node_scalar + scalar_out
        node_vector = node_vector + vector_out

        delta_scalar, delta_vector = self.node_update(node_scalar, node_vector)

        node_scalar = node_scalar + delta_scalar
        node_vector = node_vector + delta_vector

        return node_scalar, node_vector, edge_feats
