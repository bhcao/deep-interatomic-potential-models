# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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

from e3nn_jax import scatter_mean
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.initializers import uniform
from flax.typing import Dtype

from dipm.data.dataset_info import DatasetInfo
from dipm.layers import (
    get_rbf_cls,
    get_radial_envelope_cls,
)
from dipm.layers.escn import (
    MappingCoefficients,
    SO3Rotation,
    SO3Grid,
    SO3LinearV2,
    WignerMatrices,
    LayerNormType,
    EdgeDegreeEmbedding,
    get_layernorm_layer,
    mapping_coefficients,
)
from dipm.models.force_model import ForceModel, PrecallInterface
from dipm.models.atomic_energies import get_atomic_energies
from dipm.models.uma.blocks import (
    ChargeSpinDatasetEmbed,
    Edgewise,
    SpectralAtomwise,
    GridAtomwise,
    ActivationType,
    FeedForwardType,
)
from dipm.models.uma.config import UMAConfig
from dipm.utils.safe_norm import safe_norm
from dipm.typing import get_dtype


class UMA(ForceModel, PrecallInterface):
    """The UMA model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Brandon M. Wood, Misko Dzamba, Xiang Fu, Meng Gao, Muhammed Shuaibi, Luis Barroso-Luque,
          Kareem Abdelmaqsoud, Vahe Gharakhanyan, John R. Kitchin, Daniel S. Levine, etc.
          UMA: A Family of Universal Models for Atoms. arXiv, Jun 2025.
          URL: https://arxiv.org/abs/2506.23971.

    Attributes:
        config: Hyperparameters / configuration for the UMA model, see
                :class:`~dipm.models.uma.config.UMAConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = UMAConfig
    config: UMAConfig
    force_head_prefix = "force_head"

    def __init__(
        self,
        config: dict | UMAConfig,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
        rngs: nnx.Rngs
    ):
        super().__init__(config, dataset_info, dtype=dtype)
        dtype = self.dtype
        param_dtype = get_dtype(self.config.param_dtype)

        r_max = self.dataset_info.cutoff_distance_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        uma_kargs = dict(
            num_layers=self.config.num_layers,
            lmax=self.config.lmax,
            mmax=self.config.mmax,
            sphere_channels=self.config.sphere_channels,
            edge_channels=self.config.edge_channels,
            hidden_channels=self.config.hidden_channels,
            num_rbf=self.config.num_rbf,
            grid_resolution=self.config.grid_resolution,
            norm_type=self.config.norm_type,
            act_type=self.config.act_type,
            ff_type=self.config.ff_type,
            num_experts=self.config.num_experts,
            cutoff=r_max,
            num_species=num_species,
        )

        self.charge_spin_dataset_embed = ChargeSpinDatasetEmbed(
            self.config.sphere_channels,
            None if self.config.dataset_list is None else len(self.config.dataset_list),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if self.config.num_experts > 0:
            if self.config.use_composition_embedding:
                router_channels = 2 * self.config.sphere_channels
                self.composition_embedding = nnx.Embed(
                    num_species, self.config.sphere_channels,
                    dtype=dtype, param_dtype=param_dtype, rngs=rngs
                )
            else:
                router_channels = self.config.sphere_channels
            num_experts = self.config.num_experts

            self.mole_router = nnx.Sequential(
                nnx.Linear(router_channels, num_experts * 2,
                           dtype=dtype, param_dtype=param_dtype, rngs=rngs),
                nnx.silu,
                nnx.Linear(num_experts * 2, num_experts * 2,
                           dtype=dtype, param_dtype=param_dtype, rngs=rngs),
                nnx.silu,
                nnx.Linear(num_experts * 2, num_experts,
                           dtype=dtype, param_dtype=param_dtype, rngs=rngs),
                nnx.silu,
            )

            self.mole_dropout = nnx.Dropout(self.config.mole_dropout)

        self.backbone = UMABlock(
            **uma_kargs,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.energy_head = nnx.Sequential(
            nnx.Linear(
                self.config.sphere_channels, self.config.hidden_channels,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            ),
            nnx.silu,
            nnx.Linear(
                self.config.hidden_channels, self.config.hidden_channels,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            ),
            nnx.silu,
            nnx.Linear(
                self.config.hidden_channels, 1, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            ),
        )

        if self.predict_forces:
            self.force_head = SO3LinearV2(
                self.config.sphere_channels, 1, lmax=1,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

        self.atomic_energies = nnx.Cache(get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species, dtype=dtype
        ))

    # pylint: disable=arguments-differ
    def precall(
        self,
        node_species: jax.Array, # [num_nodes,]
        charge: jax.Array, # [num_batch,]
        spin: jax.Array, # [num_batch,]
        n_node: jax.Array, # [num_batch,]
        dataset: jax.Array | None = None,
        rngs: nnx.Rngs | None = None,
        **_kwargs,
    ) -> dict:
        if dataset is None:
            if self.config.dataset_list is not None:
                raise ValueError("Must provide `dataset` for `precall`.")
            csd_mixed_emb = self.charge_spin_dataset_embed(charge, spin)
        else:
            csd_mixed_emb = self.charge_spin_dataset_embed(charge, spin, dataset)

        # Just cache the csd_mixed_emb, no futher precall.
        if self.config.num_experts == 0:
            return {".": {"csd_mixed_emb": csd_mixed_emb}}

        embeddings = csd_mixed_emb
        if self.config.use_composition_embedding:
            composition = scatter_mean(self.composition_embedding(node_species), nel=n_node)
            embeddings = jnp.concat([composition, csd_mixed_emb], axis=-1)

        expert_mixing_coeffs = self.mole_router(embeddings)
        expert_mixing_coeffs = nnx.softmax(
            self.mole_dropout(expert_mixing_coeffs, rngs=rngs), axis=1
        ) + 0.005 # [batch, num_experts]

        return super().precall(
            csd_mixed_emb=csd_mixed_emb,
            expert_mixing_coeffs=expert_mixing_coeffs,
            n_node=n_node,
        )

    def cache(self, csd_mixed_emb: jax.Array, **_kwargs):
        return {"csd_mixed_emb": csd_mixed_emb}

    @PrecallInterface.context_handler
    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        n_node: jax.Array, # Nel version of pyg.Data.batch
        rngs: nnx.Rngs | None = None, # Rngs for dropout, None for eval
        *,
        csd_mixed_emb: jax.Array,
        ctx: dict,
    ) -> jax.Array:
        node_feats = self.backbone(
            edge_vectors, node_species, csd_mixed_emb, senders, receivers, n_node, rngs, ctx=ctx
        )

        node_energies = self.energy_head(node_feats[:, 0])[:, 0]

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        node_energies += self.atomic_energies.value[node_species]  # [n_nodes, ]

        if self.predict_forces:
            forces = self.force_head(node_feats[:, :4])[:, 1:4, 0]
            return node_energies, forces

        return node_energies


class UMABlock(nnx.Module, PrecallInterface):
    def __init__(
        self,
        num_species: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        num_rbf: int = 512,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: LayerNormType = LayerNormType.LAYER_NORM_SH,
        act_type: ActivationType = ActivationType.GATE,
        ff_type: FeedForwardType = FeedForwardType.GRID,
        num_experts: int = 0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.sphere_channels = sphere_channels
        self.cutoff = cutoff
        self.lmax = lmax
        self.deterministic = False # Randomness

        mapping_coeffs = mapping_coefficients(lmax, mmax)

        # lmax_lmax for node, lmax_mmax for edge
        so3_grid = SO3Grid(lmax, mmax, resolution=grid_resolution, dtype=dtype or param_dtype)
        so3_grid_lmax = SO3Grid(lmax, lmax, resolution=grid_resolution, dtype=dtype or param_dtype)

        # atom embedding
        self.sphere_embedding = nnx.Embed(
            num_species, sphere_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # edge distance embedding
        self.distance_expansion = get_rbf_cls("gauss")(
            cutoff,
            num_rbf,
            rbf_width=2.0,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # equivariant initial embedding
        self.senders_embedding = nnx.Embed(
            num_species, edge_channels, embedding_init=uniform(0.001), # Why?
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.receivers_embedding = nnx.Embed(
            num_species, edge_channels, embedding_init=uniform(0.001),
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        edge_channels_list = [
            num_rbf + 2 * edge_channels,
            edge_channels,
            edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels,
            mapping_coeffs,
            edge_channels_list,
            use_atom_edge_embedding=False,
            rescale_factor=5.0,  # sqrt avg degree
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.envelope = get_radial_envelope_cls("polynomial_envelope")(self.cutoff)

        # Initialize the blocks for each layer
        self.layers = []
        for _ in range(num_layers):
            layer = UMALayer(
                sphere_channels,
                hidden_channels,
                mapping_coeffs,
                so3_grid,
                so3_grid_lmax,
                edge_channels_list,
                norm_type,
                act_type,
                ff_type,
                num_experts,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.layers.append(layer)

        self.norm = get_layernorm_layer(
            norm_type,
            lmax,
            num_channels=sphere_channels,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.so3_rotation = SO3Rotation(
            lmax, mmax, mapping_coeffs.perm, scale=False, dtype=dtype or param_dtype
        )

    @PrecallInterface.context_handler
    def __call__(
        self,
        edge_vectors: jax.Array,  # [n_edges, 3]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        csd_mixed_emb: jax.Array,  # [n_batch, sphere_channels]
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        n_node: jax.Array,  # [n_batch]
        rngs: nnx.Rngs | None = None,  # Rngs for dropout, None for eval
        *,
        ctx: dict | None = None,
    ):
        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        if not self.deterministic:
            assert rngs is not None
            rot_gamma = jax.random.uniform(
                rngs['rotation'](), shape=len(edge_vectors), maxval=2 * jnp.pi,
                dtype=edge_vectors.dtype
            )
        else:
            rot_gamma = jnp.zeros(len(edge_vectors), dtype=edge_vectors.dtype)

        wigner_matrices = self.so3_rotation.create_wigner_matrices(edge_vectors, rot_gamma)

        # Init per node representations using an atomic number based embedding
        node_feats_scaler = self.sphere_embedding(node_species)
        batch = jnp.repeat(jnp.arange(len(n_node)), n_node, total_repeat_length=len(node_species))
        node_feats_scaler += csd_mixed_emb[batch]

        node_feats_pad = jnp.zeros(
            (len(node_species), (self.lmax + 1) ** 2 - 1, self.sphere_channels),
            dtype=node_feats_scaler.dtype,
        )
        node_feats = jnp.concat([node_feats_scaler[:, None], node_feats_pad], axis=1)

        # edge degree embedding
        edge_distances = safe_norm(edge_vectors, axis=-1)
        edge_envelope = self.envelope(edge_distances).reshape(-1, 1, 1)

        edge_distance_embeds = self.distance_expansion(edge_distances)
        senders_embeds = self.senders_embedding(node_species[senders])
        receivers_embeds = self.receivers_embedding(node_species[receivers])
        edge_embeds = jnp.concat(
            (edge_distance_embeds, senders_embeds, receivers_embeds), axis=1
        )
        node_feats = self.edge_degree_embedding(
            node_feats,
            edge_embeds,
            senders,
            receivers,
            wigner_matrices,
            edge_envelope,
        )

        for layer in self.layers:
            node_feats = layer(
                node_feats,
                edge_embeds,
                senders,
                receivers,
                wigner_matrices,
                edge_envelope,
                ctx=ctx,
            )

        # Final layer norm
        node_feats = self.norm(node_feats)
        return node_feats


class UMALayer(nnx.Module, PrecallInterface):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        mapping_coeffs: MappingCoefficients,
        so3_grid: SO3Grid,
        so3_grid_lmax: SO3Grid,
        edge_channels_list: list[int],
        norm_type: LayerNormType,
        act_type: ActivationType,
        ff_type: FeedForwardType,
        num_experts: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.norm_1 = get_layernorm_layer(
            norm_type, mapping_coeffs.lmax, sphere_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.edge_wise = Edgewise(
            sphere_channels,
            hidden_channels,
            edge_channels_list,
            mapping_coeffs,
            so3_grid,
            act_type=act_type,
            num_experts=num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.norm_2 = get_layernorm_layer(
            norm_type, mapping_coeffs.lmax, sphere_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        if ff_type == FeedForwardType.SPECTRAL:
            self.atom_wise = SpectralAtomwise(
                sphere_channels,
                hidden_channels,
                mapping_coeffs.lmax,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.atom_wise = GridAtomwise(
                sphere_channels,
                hidden_channels,
                so3_grid_lmax=so3_grid_lmax,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    @PrecallInterface.context_handler
    def __call__(
        self,
        node_feats: jax.Array,
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMatrices,
        edge_envelope: jax.Array,
        *,
        ctx: dict | None = None,
    ):
        # Edge-wise
        node_feats_res = node_feats
        node_feats = self.norm_1(node_feats)

        node_feats = self.edge_wise(
            node_feats,
            edge_embeds,
            senders,
            receivers,
            wigner_matrices,
            edge_envelope,
            ctx=ctx,
        )
        node_feats = node_feats + node_feats_res

        # Atom-wise
        node_feats_res = node_feats
        node_feats = self.norm_2(node_feats)

        node_feats = self.atom_wise(node_feats)
        node_feats = node_feats + node_feats_res

        return node_feats
