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

# flake8: noqa: N806
from abc import ABCMeta, abstractmethod
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from flax.linen import initializers

from dipm.layers import (
    CosineCutoff,
    get_activation_fn,
)


class Sphere(nnx.Module):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def __call__(self, edge_vec: jax.Array) -> jax.Array:
        edge_sh = self._spherical_harmonics(
            edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )
        return edge_sh

    def _spherical_harmonics(
        self, x: jax.Array, y: jax.Array, z: jax.Array
    ) -> jax.Array:
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if self.degree == 1:
            return jnp.stack([sh_1_0, sh_1_1, sh_1_2], axis=-1)

        sh_2_0 = jnp.sqrt(3.0) * x * z
        sh_2_1 = jnp.sqrt(3.0) * x * y
        y2 = y**2
        x2z2 = x**2 + z**2
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = jnp.sqrt(3.0) * y * z
        sh_2_4 = jnp.sqrt(3.0) / 2.0 * (z**2 - x**2)

        if self.degree == 2:
            return jnp.stack(
                [sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
                axis=-1,
            )


class NeighborEmbedding(nnx.Module):
    def __init__(
        self,
        num_rbf: int,
        num_channels: int,
        cutoff: float,
        num_species: int = 100,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.embedding = nnx.Embed(
            num_species, num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.distance_proj = nnx.Linear(
            num_rbf,
            num_channels,
            kernel_init=initializers.xavier_uniform(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.combine = nnx.Linear(
            2 * num_channels,
            num_channels,
            kernel_init=initializers.xavier_uniform(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)

    def __call__(self, node_z, node_feats, senders, receivers, edge_weight, edge_feats):

        C = self.cutoff_fn(edge_weight)
        W = self.distance_proj(edge_feats) * C[:, None] # [n_edges, num_channels]

        x_neighbors = self.embedding(node_z)
        x_j = x_neighbors[senders] # [n_edges, num_channels]

        # message function
        node_msgs = x_j * W
        aggregated_msgs = jax.ops.segment_sum(
            node_msgs, receivers, num_segments=node_feats.shape[0]
        )

        # Update between x and aggregated_messages over neighbors
        node_feats = self.combine(
            jnp.concatenate([node_feats, aggregated_msgs], axis=1)
        )
        return node_feats


class EdgeEmbedding(nnx.Module):
    def __init__(
        self,
        num_rbf: int,
        num_channels: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.edge_proj = nnx.Linear(
            num_rbf,
            num_channels,
            kernel_init=initializers.xavier_uniform(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, senders, receivers, edge_attr, x):
        x_j = x[senders]
        x_i = x[receivers]

        # message function
        edge_messages = (x_i + x_j) * self.edge_proj(edge_attr)
        return edge_messages


class GatedEquivariantBlock(nnx.Module):
    def __init__(
        self,
        feat_channels: int,
        vec_channels: int,
        num_channels: int,
        out_channels: int,
        intermediate_channels: int | None = None,
        activation: str = "silu",
        scalar_activation: bool = False,
        ignore_vec_output: bool = False,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if intermediate_channels is None:
            intermediate_channels = num_channels

        # merged vec1_proj and vec2_proj
        self.vec_proj = nnx.Linear(
            vec_channels,
            num_channels if ignore_vec_output else num_channels + out_channels,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.update_net = nnx.Sequential(
            nnx.Linear(
                feat_channels + vec_channels,
                intermediate_channels,
                kernel_init=initializers.xavier_uniform(),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            get_activation_fn(activation),
            nnx.Linear(
                intermediate_channels,
                out_channels if ignore_vec_output else out_channels * 2,
                kernel_init=initializers.xavier_uniform(),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
        )

        if scalar_activation:
            # Assuming direct call is intended, otherwise needs adjustment
            self.act = get_activation_fn(activation)

        self.scalar_activation = scalar_activation
        self.ignore_vec_output = ignore_vec_output
        self.num_channels = num_channels

    def __call__(self, x, v):
        if self.ignore_vec_output:
            vec1 = self.vec_proj(v)
        else:
            vec1, vec2 = jnp.split(self.vec_proj(v), [self.num_channels], axis=-1)
        vec1 = jnp.linalg.norm(vec1 + 1e-8, axis=-2)
        x = jnp.concatenate([x, vec1], axis=-1) # [n_nodes, 2 * num_channels]

        x = self.update_net(x)
        if not self.ignore_vec_output:
            x, v = jnp.split(x, 2, axis=-1)

        if self.scalar_activation:
            x = self.act(x) # [n_nodes, out_channels]

        if self.ignore_vec_output:
            return x

        # pylint: disable=E0606
        v = v[:, None] * vec2 # [n_nodes, 3, out_channels]
        return x, v


class OutputModel(nnx.Module, metaclass=ABCMeta):
    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        pass  # Must be implemented in subclasses

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(
        self,
        feat_channels: int,
        num_channels: int,
        activation: str = "silu",
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.output_network = nnx.Sequential(
            nnx.Linear(
                feat_channels,
                num_channels // 2,
                kernel_init=initializers.xavier_uniform(),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            get_activation_fn(activation),
            nnx.Linear(
                num_channels // 2,
                1,
                kernel_init=initializers.xavier_uniform(),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
        )

    def pre_reduce(self, x, v, z=None, pos=None, batch=None):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    def __init__(
        self,
        feat_channels: int,
        vec_channels: int,
        num_channels: int,
        activation: str = "silu",
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.output_network = [
            GatedEquivariantBlock(
                feat_channels,
                vec_channels,
                num_channels,
                num_channels // 2,
                activation=activation,
                scalar_activation=True,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            GatedEquivariantBlock(
                num_channels // 2,
                num_channels // 2,
                num_channels // 2,
                1,
                activation=activation,
                scalar_activation=False,
                ignore_vec_output=True,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
        ]

    def pre_reduce(self, x, v, z=None, pos=None, batch=None):
        x, v = self.output_network[0](x, v)
        return self.output_network[1](x, v)
