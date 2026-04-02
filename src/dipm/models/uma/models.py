# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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

import re

from e3nn_jax import scatter_mean
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.initializers import uniform
from flax.typing import Dtype

from dipm.data.dataset_info import DatasetInfo
from dipm.layers.radial_basis import GaussianBasis
from dipm.layers.cutoff import PolynomialCutoff
from dipm.layers.escn import (
    get_wigner_mats,
    SO3LinearV2,
    WignerMats,
    LayerNormType,
    EdgeDegreeEmbedding,
    get_layernorm_layer,
)
from dipm.models.force_model import ForceModel
from dipm.models.atomic_energies import get_atomic_energies
from dipm.models.uma.blocks import (
    ChargeSpinTaskEmbed,
    Edgewise,
    SpectralAtomwise,
    GridAtomwise,
    ActivationType,
    FeedForwardType,
)
from dipm.models.uma.config import UMAConfig
from dipm.utils.safe_norm import safe_norm


class UMA(ForceModel):
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
    embedding_layer_regexp = re.compile(
        r"\.(composition|sphere|senders|receivers)_embedding\.embedding$"
    )

    def __init__(
        self,
        config: dict | UMAConfig,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(42)
        super().__init__(config, dataset_info, dtype=dtype)

        r_max = self.dataset_info.cutoff_distance_angstrom

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

        self.charge_spin_task_embed = ChargeSpinTaskEmbed(
            self.config.sphere_channels,
            None if self.dataset_info.task_list is None else len(self.dataset_info.task_list),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        if self.config.num_experts > 0:
            if self.config.use_composition_embedding:
                router_channels = 2 * self.config.sphere_channels
                self.composition_embedding = nnx.Embed(
                    num_species, self.config.sphere_channels,
                    dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
                )
            else:
                router_channels = self.config.sphere_channels
            num_experts = self.config.num_experts

            self.mole_router = nnx.Sequential(
                nnx.Linear(router_channels, num_experts * 2,
                           dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs),
                nnx.silu,
                nnx.Linear(num_experts * 2, num_experts * 2,
                           dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs),
                nnx.silu,
                nnx.Linear(num_experts * 2, num_experts,
                           dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs),
                nnx.silu,
            )

            self.mole_dropout = nnx.Dropout(self.config.mole_dropout)

            self.decode = False
            self.cached_csd_mixed_emb = nnx.data(None)

        self.backbone = UMABlock(
            **uma_kargs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs
        )

        self.energy_head = nnx.Sequential(
            nnx.Linear(
                self.config.sphere_channels, self.config.hidden_channels,
                dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            ),
            nnx.silu,
            nnx.Linear(
                self.config.hidden_channels, self.config.hidden_channels,
                dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            ),
            nnx.silu,
            nnx.Linear(
                self.config.hidden_channels, 1,
                dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            ),
        )

        if self.config.force_head:
            self.force_head = SO3LinearV2(
                self.config.sphere_channels, 1, lmax=1,
                dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            )

        self.num_species = num_species

    def precall(
        self,
        node_species: jax.Array, # [num_nodes,]
        charge: jax.Array, # [num_batch,]
        spin: jax.Array, # [num_batch,]
        n_node: jax.Array, # [num_batch,]
        task: jax.Array | None = None,
        rngs: nnx.Rngs | None = None,
        **_kwargs,
    ) -> dict:
        """Precall function that returns a dictionary of values which will be used as
        the kwargs of ``nnx.view`` to initialize the cache."""

        csd_mixed_emb, expert_coeffs = self._router_forward(
            node_species, charge, spin, n_node, task, rngs
        )

        if self.config.num_experts == 0:
            return {"csd_mixed_emb": csd_mixed_emb}

        return {
            "csd_mixed_emb": csd_mixed_emb,
            "expert_coeffs": expert_coeffs,
            "n_node": n_node,
        }

    def _router_forward(
        self,
        node_species: jax.Array, # [num_nodes,]
        charge: jax.Array, # [num_batch,]
        spin: jax.Array, # [num_batch,]
        n_node: jax.Array, # [num_batch,]
        task: jax.Array | None = None,
        rngs: nnx.Rngs | None = None,
        **_kwargs,
    ) -> tuple[jax.Array, jax.Array | None]:
        if task is None and self.dataset_info.task_list is not None:
            raise ValueError("Must provide `task` for `precall`.")
        if self.dataset_info.task_list is None:
            csd_mixed_emb = self.charge_spin_task_embed(charge, spin)
        else:
            csd_mixed_emb = self.charge_spin_task_embed(charge, spin, task)

        if self.config.num_experts == 0:
            return csd_mixed_emb, None

        embeddings = csd_mixed_emb
        if self.config.use_composition_embedding:
            composition = scatter_mean(
                self.composition_embedding(node_species), nel=n_node
            )
            embeddings = jnp.concat([composition, csd_mixed_emb], axis=-1)

        expert_coeffs = self.mole_router(embeddings)
        expert_coeffs = nnx.softmax(
            self.mole_dropout(expert_coeffs, rngs=rngs), axis=1
        ) + 0.005 # [batch, num_experts]

        return csd_mixed_emb, expert_coeffs

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        *,
        charge: jax.Array,
        spin: jax.Array,
        n_node: jax.Array, # Nel version of pyg.Data.batch
        task: jax.Array | None,
        rngs: nnx.Rngs | None = None, # Rngs for dropout, None for eval
        **_kwargs,
    ) -> jax.Array:
        if self.decode:
            assert self.cached_csd_mixed_emb is not None, (
                "nnx.view must be called before performing inference"
            )
            csd_mixed_emb = self.cached_csd_mixed_emb.value
            expert_coeffs = None
        else:
            csd_mixed_emb, expert_coeffs = self._router_forward(
                node_species, charge, spin, n_node, task, rngs
            )

        node_feats = self.backbone(
            edge_vectors, node_species, csd_mixed_emb, senders, receivers, n_node,
            rngs=rngs, expert_coeffs=expert_coeffs,
        )

        node_energies = self.energy_head(node_feats[:, 0])[:, 0]

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        atomic_energies = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, self.num_species, self.dtype
        )
        if self.dataset_info.task_list is not None:
            task = jnp.repeat(task, n_node, total_repeat_length=len(node_species))
            node_energies += atomic_energies[node_species, task]  # [n_nodes, ]
        else:
            node_energies += atomic_energies[node_species]  # [n_nodes, ]

        if self.config.force_head:
            forces = self.force_head(node_feats[:, :4])[:, 1:4, 0]
            return node_energies, std * forces

        return node_energies

    def set_view(
        self,
        decode: bool | None = None,
        csd_mixed_emb: jax.Array | None = None,
        **kwargs
    ):
        """Class method used by ``nnx.view``.
        
        Args:
            decode: If True, the module is set to decode mode.
            csd_mixed_emb: The charge-spin-dataset mixed embedding.
        """

        if decode is not None:
            self.decode = decode

            if decode:
                assert csd_mixed_emb is not None, (
                    "csd_mixed_emb must be provided in decode mode"
                )
                self.cached_csd_mixed_emb = nnx.Cache(csd_mixed_emb)

        return kwargs


class UMABlock(nnx.Module):
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
        self.mmax = mmax
        self.deterministic = False # Randomness

        # atom embedding
        self.sphere_embedding = nnx.Embed(
            num_species, sphere_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # edge distance embedding
        self.distance_expansion = GaussianBasis(
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
            lmax,
            mmax,
            sphere_channels,
            edge_channels_list,
            use_atom_edge_embedding=False,
            rescale_factor=5.0,  # sqrt avg degree
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.envelope = PolynomialCutoff(self.cutoff)

        # Initialize the blocks for each layer
        self.layers = nnx.List([
            UMALayer(
                lmax,
                mmax,
                sphere_channels,
                hidden_channels,
                grid_resolution,
                edge_channels_list,
                norm_type,
                act_type,
                ff_type,
                num_experts,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        self.norm = get_layernorm_layer(
            norm_type,
            lmax,
            num_channels=sphere_channels,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        edge_vectors: jax.Array,  # [n_edges, 3]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        csd_mixed_emb: jax.Array,  # [n_batch, sphere_channels]
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        n_node: jax.Array,  # [n_batch]
        *,
        rngs: nnx.Rngs | None = None,  # Rngs for dropout, None for eval
        expert_coeffs: jax.Array | None = None,
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

        wigner_matrices = get_wigner_mats(
            self.lmax, self.mmax, edge_vectors, rot_gamma, scale=False
        )

        # Init per node representations using an atomic number based embedding
        node_feats_scaler = self.sphere_embedding(node_species)
        csd_mixed_emb = jnp.repeat(
            csd_mixed_emb, n_node, total_repeat_length=len(node_species), axis=0
        )
        node_feats_scaler += csd_mixed_emb

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
                n_node=n_node,
                expert_coeffs=expert_coeffs,
            )

        # Final layer norm
        node_feats = self.norm(node_feats)
        return node_feats


class UMALayer(nnx.Module):
    def __init__(
        self,
        lmax: int,
        mmax: int,
        sphere_channels: int,
        hidden_channels: int,
        grid_resolution: int,
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
            norm_type, lmax, sphere_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.edge_wise = Edgewise(
            lmax,
            mmax,
            sphere_channels,
            hidden_channels,
            edge_channels_list,
            grid_resolution,
            act_type=act_type,
            num_experts=num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.norm_2 = get_layernorm_layer(
            norm_type, lmax, sphere_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        if ff_type == FeedForwardType.SPECTRAL:
            self.atom_wise = SpectralAtomwise(
                sphere_channels,
                hidden_channels,
                lmax,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.atom_wise = GridAtomwise(
                sphere_channels,
                hidden_channels,
                lmax,
                grid_resolution,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(
        self,
        node_feats: jax.Array,
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_matrices: WignerMats,
        edge_envelope: jax.Array,
        *,
        n_node: jax.Array | None = None,
        expert_coeffs: jax.Array | None = None,
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
            n_node=n_node,
            expert_coeffs=expert_coeffs,
        )
        node_feats = node_feats + node_feats_res

        # Atom-wise
        node_feats_res = node_feats
        node_feats = self.norm_2(node_feats)

        node_feats = self.atom_wise(node_feats)
        node_feats = node_feats + node_feats_res

        return node_feats
