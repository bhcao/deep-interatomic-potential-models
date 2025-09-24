# Copyright 2025 InstaDeep Ltd and Cao Bohan
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
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Dtype

from dipm.data.dataset_info import DatasetInfo
from dipm.models.atomic_energies import get_atomic_energies
from dipm.models.force_model import ForceModel
from dipm.layers import (
    CosineCutoff,
    get_activation_fn,
    get_veclayernorm_fn,
    get_rbf_cls,
)
from dipm.models.visnet.blocks import (
    EdgeEmbedding,
    EquivariantScalar,
    NeighborEmbedding,
    Sphere,
)
from dipm.models.visnet.config import VisnetConfig
from dipm.utils.safe_norm import safe_norm


class Visnet(ForceModel):
    """The ViSNet model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Yusong Wang, Tong Wang, Shaoning Li, Xinheng He, Mingyu Li, Zun Wang,
          Nanning Zheng, Bin Shao, and Tie-Yan Liu. Enhancing geometric
          representations for molecules with equivariant vector-scalar interactive
          message passing. Nature Communications, 15(1), January 2024.
          ISSN: 2041-1723. URL: https://dx.doi.org/10.1038/s41467-023-43720-2.


    Attributes:
        config: Hyperparameters / configuration for the ViSNet model, see
                :class:`~dipm.models.visnet.config.VisnetConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = VisnetConfig
    config: VisnetConfig

    def __init__(
        self,
        config: dict | VisnetConfig,
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

        visnet_kwargs = dict(
            lmax=self.config.l_max,
            vecnorm_type=self.config.vecnorm_type,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            rbf_type="expnorm",
            trainable_rbf=self.config.trainable_rbf,
            activation=self.config.activation,
            attn_activation=self.config.attn_activation,
            cutoff=r_max,
            num_species=num_species,
        )

        self.visnet_model = VisnetBlock(**visnet_kwargs, param_dtype=param_dtype, rngs=rngs)

        self.output_model = EquivariantScalar(
            self.config.num_channels,
            self.config.num_channels,
            self.config.num_channels,
            activation=self.config.activation,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.atomic_energies = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )

    def __call__(
        self,
        edge_vectors: jnp.ndarray,
        node_species: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> jnp.ndarray:

        node_feats, vector_feats = self.visnet_model(
            edge_vectors, node_species, senders, receivers
        )
        node_feats = self.output_model.pre_reduce(node_feats, vector_feats, node_species)
        node_feats = node_feats.squeeze(axis=-1)

        node_feats *= self.dataset_info.scaling_stdev
        node_feats = self.output_model.post_reduce(node_feats)

        node_feats += self.dataset_info.scaling_mean
        node_feats += self.atomic_energies[node_species]  # [n_nodes, ]

        return node_feats


class VisnetBlock(nnx.Module):
    def __init__(
        self,
        lmax: int = 2,
        vecnorm_type: str = "none",
        num_heads: int = 8,
        num_layers: int = 9,
        num_channels: int = 256,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        attn_activation: str = "silu",
        cutoff: float = 5.0,
        num_species: int = 5,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.vec_dim = (lmax + 1) ** 2 - 1

        assert num_channels % num_heads == 0, (
            f"The number of hidden channels ({num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.node_embedding = nnx.Embed(
            num_species, num_channels, param_dtype=param_dtype, rngs=rngs
        )
        self.radial_embedding = get_rbf_cls(rbf_type)(
            cutoff, num_rbf, trainable_rbf, param_dtype=param_dtype, rngs=rngs
        )
        self.spherical_embedding = Sphere(lmax)

        self.neighbor_embedding = NeighborEmbedding(
            num_rbf, num_channels, cutoff, num_species, param_dtype=param_dtype, rngs=rngs
        )

        self.edge_embedding = EdgeEmbedding(
            num_rbf, num_channels, param_dtype=param_dtype, rngs=rngs
        )

        self.visnet_layers = [
            VisnetLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                activation=activation,
                attn_activation=attn_activation,
                cutoff=cutoff,
                vecnorm_type=vecnorm_type,
                last_layer=i == num_layers - 1,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for i in range(num_layers)
        ]

        self.out_norm = nnx.LayerNorm(
            num_channels, epsilon=1e-05, param_dtype=param_dtype, rngs=rngs
        )
        self.vec_out_norm = get_veclayernorm_fn(vecnorm_type)

    def __call__(
        self,
        edge_vectors: jnp.ndarray,  # [n_edges, 3]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        assert edge_vectors.ndim == 2 and edge_vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert edge_vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        # Calculate distances
        distances = safe_norm(edge_vectors, axis=-1)

        # Embedding Layers
        node_feats = self.node_embedding(node_species)  # Is that necessary?

        # Seems like doubled from within the neighbor embedding module
        edge_feats = self.radial_embedding(distances)

        spherical_feats = self.spherical_embedding(
            edge_vectors / (distances[:, None] + 1e-8)
        )
        node_feats = self.neighbor_embedding(
            node_species, node_feats, senders, receivers, distances, edge_feats
        )  # h in paper

        edge_feats = self.edge_embedding(
            senders, receivers, edge_feats, node_feats
        )  # f in paper

        vec_shape = (
            node_feats.shape[0],
            self.vec_dim,
            node_feats.shape[1],
        )
        vector_feats = jnp.zeros(vec_shape, dtype=node_feats.dtype)

        for layer in self.visnet_layers:
            diff_node_feats, diff_edge_feats, diff_vector_feats = layer(
                node_feats,
                edge_feats,
                vector_feats,
                distances,
                senders,
                receivers,
                spherical_feats,
            )

            node_feats += diff_node_feats
            edge_feats += diff_edge_feats
            vector_feats += diff_vector_feats

        node_feats = self.out_norm(node_feats)
        vector_feats = self.vec_out_norm(vector_feats)
        return node_feats, vector_feats


class VisnetLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        activation: str,
        attn_activation: str,
        cutoff: float,
        vecnorm_type: str,
        last_layer: bool = False,
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.last_layer = last_layer
        self.head_dim = num_channels // num_heads

        # Setting eps=1e-05 to reproduce pytorch Layernorm
        # See: https://github.com/cgarciae/nanoGPT-jax/blob/24fd60f987a946915e43c0000195bd73ddc34271/model.py#L95  # noqa: E501
        self.layernorm = nnx.LayerNorm(
            num_channels, epsilon=1e-05, param_dtype=param_dtype, rngs=rngs
        )
        self.vec_layernorm = get_veclayernorm_fn(vecnorm_type)
        self.act = get_activation_fn(activation)
        self.attn_act = get_activation_fn(attn_activation)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.vec_proj = nnx.Linear(
            num_channels,
            num_channels * 3,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.kqv_proj = nnx.Linear(
            num_channels,
            3 * num_channels,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dkdv_proj = nnx.Linear(
            num_channels,
            2 * num_channels,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.s_proj = nnx.Linear(
            num_channels,
            num_channels * 2,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            num_channels,
            num_channels * 3,
            kernel_init=initializers.xavier_uniform(),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if not last_layer:
            self.f_proj = nnx.Linear(
                num_channels,
                num_channels,
                kernel_init=initializers.xavier_uniform(),
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.w_src_proj = nnx.Linear(
                num_channels,
                num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.w_trg_proj = nnx.Linear(
                num_channels,
                num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def message_fn(
        self,
        q_i: jnp.ndarray,
        k_j: jnp.ndarray,
        v_j: jnp.ndarray,
        vec_j: jnp.ndarray,
        dk: jnp.ndarray,
        dv: jnp.ndarray,
        r_ij: jnp.ndarray,
        d_ij: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        attn = (q_i * k_j * dk).sum(axis=-1)
        attn = self.attn_act(attn) * self.cutoff_fn(r_ij)[:, None]

        v_j = v_j * dv
        v_j = (v_j * attn[..., None]).reshape(-1, self.num_channels)

        s1, s2 = jnp.split(self.act(self.s_proj(v_j)), [self.num_channels], axis=1)
        vec_j = vec_j * s1[:, None] + s2[:, None] * d_ij[..., None]

        return v_j, vec_j

    def edge_update(
        self,
        vec_i: jnp.ndarray,
        vec_j: jnp.ndarray,
        d_ij: jnp.ndarray,
        f_ij: jnp.ndarray,
    ) -> jnp.ndarray:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(axis=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def __call__(
        self,
        node_feats: jnp.ndarray, # [n_nodes, num_channels]
        edge_feats: jnp.ndarray, # [n_edges, num_channels]
        vector_feats: jnp.ndarray, # [n_edges, (lmax + 1) ^ 2 - 1, num_channels]
        distances: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        sh_feats: jnp.ndarray, # [n_edges, (lmax + 1) ^ 2 - 1]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        node_feats = self.layernorm(node_feats)
        vector_feats = self.vec_layernorm(vector_feats)
        # Correspond to Wk, Wq, Wv weights in the paper
        k_feats, q_feats, v_feats = jnp.split(self.kqv_proj(node_feats), 3, axis=-1)
        # Correspond to Dk, Dv weights in the paper
        dk_feats, dv_feats = jnp.split(self.dkdv_proj(edge_feats), 2, axis=-1)
        # Reshape the outputs to include the num_heads dimension
        new_shape = (-1, self.num_heads, self.head_dim)
        q_i = jnp.reshape(q_feats, new_shape)[receivers]
        k_j = jnp.reshape(k_feats, new_shape)[senders]
        v_j = jnp.reshape(v_feats, new_shape)[senders]
        dk_feats = jnp.reshape(self.act(dk_feats), new_shape)
        dv_feats = jnp.reshape(self.act(dv_feats), new_shape)

        vec1, vec2, vec3 = jnp.split(self.vec_proj(vector_feats), 3, axis=-1)
        vec_dot = jnp.sum(vec1 * vec2, axis=1)

        # Apply message function for each edge
        vec_j = vector_feats[senders]

        node_msgs, vec_msgs = self.message_fn(
            q_i, k_j, v_j, vec_j, dk_feats, dv_feats, distances, sh_feats
        )
        # Aggregate the messages
        node_feats = jax.ops.segment_sum(
            node_msgs, receivers, num_segments=node_feats.shape[0]
        )
        vec_out = jax.ops.segment_sum(
            vec_msgs, receivers, num_segments=node_feats.shape[0]
        )

        o1, o2, o3 = jnp.split(self.o_proj(node_feats), 3, axis=1)

        dx = vec_dot * o2 + o3
        dvec = vec3 * o1[:, None] + vec_out

        if self.last_layer:
            df_ij = jnp.zeros_like(edge_feats)
        else:
            df_ij = self.edge_update(
                vector_feats[receivers], vec_j, sh_feats, edge_feats
            )
        return dx, df_ij, dvec

    def vector_rejection(self, vec, d_ij):
        # Implement vector rejection logic using JAX
        vec_proj = (vec * d_ij[..., None]).sum(axis=1, keepdims=True)
        return vec - vec_proj * d_ij[..., None]
