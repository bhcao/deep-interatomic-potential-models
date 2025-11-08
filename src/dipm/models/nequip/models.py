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

import functools
from collections.abc import Callable
import logging

import e3nn_jax as e3nn
from flax import nnx
from flax.typing import Dtype
import jax
import jax.numpy as jnp
from e3nn_jax.legacy import FunctionalTensorProduct

from dipm.data.dataset_info import DatasetInfo
from dipm.models.atomic_energies import get_atomic_energies
from dipm.layers import (
    Linear,
    MultiLayerPerceptron,
    FullyConnectedTensorProduct,
    RadialEmbeddingLayer,
    get_activation_fn,
    get_radial_envelope_cls,
    get_radial_basis_fn,
)
from dipm.models.force_model import ForceModel
from dipm.models.nequip.config import NequipConfig
from dipm.models.nequip.nequip_helpers import prod, tp_path_exists
from dipm.utils.safe_norm import safe_norm
from dipm.typing import get_dtype

logger = logging.getLogger("dipm")


class Nequip(ForceModel):
    """The NequIP model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger,
          Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt,
          and Boris Kozinsky. E(3)-equivariant graph neural networks for data-efficient
          and accurate interatomic potentials. Nature Communications, 13(1), May 2022.
          ISSN: 2041-1723. URL: https://dx.doi.org/10.1038/s41467-022-29939-5.

    Attributes:
        config: Hyperparameters / configuration for the NequIP model, see
                :class:`~dipm.models.nequip.config.NequipConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = NequipConfig
    config: NequipConfig

    def __init__(
        self,
        config: dict | NequipConfig,
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
        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        radial_envelope_fun = get_radial_envelope_cls(self.config.radial_envelope)(r_max)

        nequip_kwargs = dict(
            avg_num_neighbors=avg_num_neighbors,
            num_layers=self.config.num_layers,
            num_species=num_species,
            node_irreps=self.config.node_irreps,
            l_max=self.config.l_max,
            num_bessel=self.config.num_bessel,
            r_max=r_max,
            radial_net_nonlinearity=self.config.radial_net_nonlinearity,
            radial_net_n_hidden=self.config.radial_net_n_hidden,
            radial_net_n_layers=self.config.radial_net_n_layers,
            use_residual_connection=True,
            nonlinearities={"e": "silu", "o": "tanh"},
            avg_r_min=None,
            radial_basis=get_radial_basis_fn("bessel"),
            radial_envelope=radial_envelope_fun,
            scalar_mlp_std=self.config.scalar_mlp_std,
        )

        self.nequip_model = NequipBlock(
            **nequip_kwargs,
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

        node_energies = self.nequip_model(edge_vectors, node_species, senders, receivers)

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies
        node_energies += self.atomic_energies.value[node_species]  # [n_nodes, ]

        return node_energies


class NequipBlock(nnx.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        num_layers: int,
        num_species: int,
        node_irreps: e3nn.Irreps | str,
        l_max: int,
        num_bessel: int,
        r_max: float,
        radial_net_nonlinearity: str,
        radial_net_n_hidden: int,
        radial_net_n_layers: int,
        scalar_mlp_std: float,
        use_residual_connection: bool,
        nonlinearities: str | dict[str, str],
        avg_r_min: float,
        radial_basis: Callable[[jax.Array], jax.Array],
        radial_envelope: Callable[[jax.Array], jax.Array],
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        self.num_species = num_species
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(l_max)

        node_irreps = e3nn.Irreps(node_irreps)

        # Non-scaler output will be trunctated
        self.node_embeddings = Linear(
            irreps_in=e3nn.Irreps(f"{num_species}x0e"),
            irreps_out=node_irreps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.radial_embeddings = RadialEmbeddingLayer(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
            num_bessel=num_bessel,
        )

        # Irreps are simplified from node_irreps to in_irreps since some are zero
        in_irreps = node_irreps.filter(keep='0e')
        self.layers = []
        for _ in range(num_layers):
            layer = NequipLayer(
                in_irreps=in_irreps,
                node_irreps=node_irreps,
                sh_irreps=self.sh_irreps,
                num_species=num_species,
                use_residual_connection=use_residual_connection,
                nonlinearities=nonlinearities,
                radial_net_nonlinearity=radial_net_nonlinearity,
                radial_net_n_hidden=radial_net_n_hidden,
                radial_net_n_layers=radial_net_n_layers,
                num_bessel=num_bessel,
                avg_num_neighbors=avg_num_neighbors,
                scalar_mlp_std=scalar_mlp_std,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            in_irreps = layer.out_irreps
            self.layers.append(layer)

        if in_irreps != node_irreps:
            logger.warning("Unnecessary node irreps: %s. Irreps used: %s.", node_irreps, in_irreps)

        # output block
        for mul, ir in node_irreps:
            if ir == e3nn.Irrep("0e"):
                mul_second_to_final = mul // 2

        second_to_final_irreps = e3nn.Irreps(f"{mul_second_to_final}x0e")  # pylint: disable=E0606
        final_irreps = e3nn.Irreps("1x0e")

        self.output = nnx.Sequential(
            Linear(in_irreps, second_to_final_irreps,
                   dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            Linear(second_to_final_irreps, final_irreps,
                   dtype=dtype, param_dtype=param_dtype, rngs=rngs),
        )

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        # Nodes Embedding
        embedding_irreps = e3nn.Irreps(f"{self.num_species}x0e")
        identity = jnp.eye(self.num_species)
        node_attr = e3nn.IrrepsArray(embedding_irreps, identity[node_species])
        node_feats = self.node_embeddings(node_attr)

        # Edges Embedding
        if hasattr(edge_vectors, "irreps"):
            edge_vectors = edge_vectors.array
        scalar_dr_edge = safe_norm(edge_vectors, axis=-1)

        edge_sh = e3nn.spherical_harmonics(self.sh_irreps, edge_vectors, normalize=True)

        embedded_dr_edge = self.radial_embeddings(scalar_dr_edge) # [n_edges, num_bessel]

        # Starting Convolution Layers
        for layer in self.layers:
            node_feats = layer(
                node_feats,
                node_attr,
                edge_sh,
                senders,
                receivers,
                embedded_dr_edge.array,
            )

        node_energies = self.output(node_feats).array
        return jnp.ravel(node_energies)


class NequipLayer(nnx.Module):
    """NequIP Convolution.

    Adapted from Google DeepMind materials discovery:
    https://github.com/google-deepmind/materials_discovery/blob/main/model/nequip.py

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.

    Args:
        node_irreps: representation of hidden/latent node-wise features
        use_residual_connection: use residual connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        num_bessel: number of Bessel basis functions to use
        avg_num_neighbors: constant number of per-atom neighbors, used for internal
          normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

    Returns:
        Updated node features h after the convolution.
    """

    def __init__(
        self,
        in_irreps: e3nn.Irreps,
        node_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        num_species: int,
        use_residual_connection: bool,
        nonlinearities: str | dict[str, str],
        radial_net_nonlinearity: str = "silu",
        radial_net_n_hidden: int = 64,
        radial_net_n_layers: int = 2,
        num_bessel: int = 8,
        avg_num_neighbors: float = 1.0,
        scalar_mlp_std: float = 4.0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.use_residual_connection = use_residual_connection
        self.avg_num_neighbors = avg_num_neighbors

        irreps_scalars = []
        irreps_nonscalars = []
        irreps_gate_scalars = []

        # get scalar target irreps
        for multiplicity, irrep in node_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if e3nn.Irrep(irrep).l == 0 and tp_path_exists(  # noqa E741
                in_irreps, sh_irreps, irrep
            ):
                irreps_scalars += [(multiplicity, irrep)]

        irreps_scalars = e3nn.Irreps(irreps_scalars)

        # get non-scalar target irreps
        for multiplicity, irrep in node_irreps:
            if e3nn.Irrep(irrep).l > 0 and tp_path_exists(
                in_irreps, sh_irreps, irrep
            ):
                irreps_nonscalars += [(multiplicity, irrep)]

        irreps_nonscalars = e3nn.Irreps(irreps_nonscalars)

        # get gate scalar irreps
        if tp_path_exists(in_irreps, sh_irreps, "0e"):
            gate_scalar_irreps_type = "0e"
        else:
            gate_scalar_irreps_type = "0o"

        for multiplicity, _irreps in irreps_nonscalars:
            irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

        irreps_gate_scalars = e3nn.Irreps(irreps_gate_scalars)

        # final layer output irreps are all three
        # note that this order is assumed by the gate function later, i.e.
        # scalars left, then gate scalar, then non-scalars
        h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

        if self.use_residual_connection:
            self.residual_tenor_product = FullyConnectedTensorProduct(
                in_irreps, e3nn.Irreps(f"{num_species}x0e"), h_out_irreps,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )

        self.linear_in = Linear(
            in_irreps, node_irreps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # gather the instructions for the tp as well as the tp output irreps
        mode = "uvu"
        trainable = "True"
        irreps_after_tp = []
        instructions = []

        for i, (mul_in1, irreps_in1) in enumerate(node_irreps.filter(keep=in_irreps)):
            for j, (_, irreps_in2) in enumerate(sh_irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in h_out_irreps:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # sort irreps to be in a l-increasing order
        irreps_after_tp, p, _ = e3nn.Irreps(irreps_after_tp).sort()

        # sort instructions
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [
                (
                    irreps_in1,
                    irreps_in2,
                    p[irreps_out],
                    mode,
                    trainable,
                )
            ]

        # TP between spherical harmonics embedding of the edge vector
        self.tensor_product = FunctionalTensorProduct(
            irreps_in1=node_irreps.filter(keep=in_irreps),
            irreps_in2=sh_irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions,
        )

        n_tp_weights = 0

        # get output dim of radial MLP / number of TP weights
        for ins in self.tensor_product.instructions:
            if ins.has_weight:
                n_tp_weights += prod(ins.path_shape)

        # The first feature is input features
        self.mlp = MultiLayerPerceptron(
            [num_bessel] + [radial_net_n_hidden] * radial_net_n_layers + [n_tp_weights],
            activation=radial_net_nonlinearity,
            use_bias=False,
            use_act_norm=False,
            gradient_normalization=0.0,
            scalar_mlp_std=scalar_mlp_std,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.linear_out = Linear(
            irreps_after_tp, h_out_irreps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # gate nonlinearity, applied to gate data, consisting of:
        # a) regular scalars,
        # b) gate scalars, and
        # c) non-scalars to be gated
        # in this order
        self.gate_fn = functools.partial(
            e3nn.gate,
            even_act=get_activation_fn(nonlinearities["e"]),
            odd_act=get_activation_fn(nonlinearities["o"]),
            even_gate_act=get_activation_fn(nonlinearities["e"]),
            odd_gate_act=get_activation_fn(nonlinearities["o"]),
        )

        # This is the out irreps for next layer to input.
        self.out_irreps = e3nn.gate(h_out_irreps)

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        node_attrs: e3nn.IrrepsArray,
        edge_sh: e3nn.IrrepsArray,
        senders: jax.Array,
        receivers: jax.Array,
        edge_embedded: jax.Array,
    ) -> e3nn.IrrepsArray:

        if self.use_residual_connection:
            res_conn = self.residual_tenor_product(node_feats, node_attrs)

        # first linear, stays in current h-space
        node_feats = self.linear_in(node_feats)

        # map node features onto edges for tp
        edge_features = node_feats[senders]

        # the TP weights (v dimension) are given by the FC
        weight = self.mlp(edge_embedded)

        edge_features = e3nn.utils.vmap(
            self.tensor_product.left_right
        )(weight, edge_features, edge_sh)

        edge_feats = edge_features.remove_zero_chunks().simplify()
        # aggregate edge features on nodes
        node_feats = e3nn.scatter_sum(
            edge_feats,
            dst=receivers,
            output_size=node_feats.shape[0],
        )

        # normalize by the average (not local) number of neighbors
        node_feats = node_feats / self.avg_num_neighbors

        # second linear, now we create extra gate scalars by mapping to h-out
        node_feats = self.linear_out(node_feats)

        if self.use_residual_connection:
            node_feats = node_feats + res_conn

        node_feats = self.gate_fn(node_feats)

        return node_feats
