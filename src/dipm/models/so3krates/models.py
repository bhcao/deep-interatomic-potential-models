# Copyright 2025 Cao Bohan
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

import e3nn_jax as e3nn
from flax import nnx
from flax.typing import Dtype
import jax
import jax.numpy as jnp

from dipm.data.dataset_info import DatasetInfo
from dipm.models.atomic_energies import get_atomic_energies
from dipm.models.force_model import ForceModel
from dipm.models.so3krates.blocks import (
    MLP, ResidualMLP, FeatureBlock, GeometricBlock, InteractionBlock, ZBLRepulsion
)
from dipm.models.so3krates.config import So3kratesConfig
from dipm.layers import (
    get_cutoff_cls,
    get_radial_basis_cls
)
from dipm.utils.safe_norm import safe_norm


class So3krates(ForceModel):
    """The So3krates model flax module. It is derived from the
    :class:`~dipm.models.force_model.ForceModel` class.

    References:
        * Frank Thorben, Oliver Unke and Klaus-Robert Müller. So3krates: Equivariant
          attention for interactions on arbitrary length-scales in molecular systems.
          Advances in Neural Information Processing Systems, 35, Dec 2022.
          URL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/bcf4ca90a8d405201d29dd47d75ac896-Abstract-Conference.html

    Attributes:
        config: Hyperparameters / configuration for the So3krates model, see
                :class:`~dipm.models.so3krates.config.So3kratesConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = So3kratesConfig
    config: So3kratesConfig
    # TODO: head support

    def __init__(
        self,
        config: dict | So3kratesConfig,
        dataset_info: DatasetInfo,
        *,
        dtype: Dtype | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(42)
        super().__init__(config, dataset_info, dtype=dtype)

        r_max = self.dataset_info.cutoff_distance_angstrom

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

        num_species = len(self.dataset_info.atomic_energies_map)

        # Is it necessary to allow users to modify here?
        chi_irreps = e3nn.Irreps(self.config.chi_irreps)
        rad_features = [self.config.num_rbf] + [self.config.num_channels] * 2
        sph_features = [
            chi_irreps.num_irreps, self.config.num_channels // 4, self.config.num_channels
        ]

        so3krates_kwargs = dict(
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            chi_irreps=chi_irreps,
            fb_rad_features=rad_features,
            gb_rad_features=rad_features,
            fb_sph_features=sph_features,
            gb_sph_features=sph_features,
            radial_basis_fn=self.config.radial_basis_fn,
            sphc_normalization=self.config.sphc_normalization,
            residual_mlp_1=self.config.residual_mlp_1,
            residual_mlp_2=self.config.residual_mlp_2,
            normalization=self.config.normalization,
            activation=self.config.activation,
            cutoff=r_max,
            num_species=num_species,
            avg_num_neighbors=avg_num_neighbors
        )

        self.backbone = So3kratesBlock(
            **so3krates_kwargs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs
        )

        self.cutoff_fn = get_cutoff_cls(self.config.radial_cutoff_fn)(r_max)

        if self.config.zbl_repulsion:
            index_to_z = tuple(sorted(self.dataset_info.atomic_energies_map.keys()))
            self.zbl_repulsion = ZBLRepulsion(
                index_to_z, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            )

        self.atomic_energies = nnx.Cache(get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species, dtype=self.dtype
        ))

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        *,
        n_node: jax.Array,
        task: jax.Array | None = None,
        charge: jax.Array | None = None, # TODO: support charge
        spin: jax.Array | None = None, # TODO: support spin
        **_kwargs,
    ) -> jax.Array:
        # This will be used by the ZBL repulsion term
        distances = safe_norm(edge_vectors, axis=-1)
        cutoffs = self.cutoff_fn(distances)

        node_energies = self.backbone(
            edge_vectors, distances, cutoffs, node_species, senders, receivers
        )
        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        if self.config.zbl_repulsion:
            e_rep = self.zbl_repulsion(
                node_species, distances, cutoffs, senders, receivers
            )
            node_energies += e_rep - self.config.zbl_repulsion_shift

        # TODO: multi-task support
        if self.dataset_info.task_list is not None:
            task = jnp.repeat(task, n_node, total_repeat_length=len(node_species))
            node_energies += self.atomic_energies.value[node_species, task]  # [n_nodes, ]
        else:
            node_energies += self.atomic_energies.value[node_species]  # [n_nodes, ]

        return node_energies


class So3kratesBlock(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        num_species: int,
        num_rbf: int,
        chi_irreps: e3nn.Irreps,
        fb_rad_features: list[int],
        gb_rad_features: list[int],
        fb_sph_features: list[int],
        gb_sph_features: list[int],
        cutoff: float = 5.0,
        radial_basis_fn: str = 'phys',
        sphc_normalization: float | None = None,
        activation: str = 'silu',
        num_heads: int = 4,
        residual_mlp_1: bool = False,
        residual_mlp_2: bool = False,
        normalization: bool = False,
        # In the original So3krates repo, this scaling factor does not exist. But for deeper networks,
        # this is necessary to ensure that the initial loss does not explode.
        avg_num_neighbors: float = 1.0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.chi_irreps = chi_irreps
        self.sphc_normalization = sphc_normalization

        self.radial_embedding = get_radial_basis_cls(radial_basis_fn)(
            cutoff, num_rbf, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.node_embedding = nnx.Embed(
            num_species, num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.layers = nnx.List([
            So3kratesLayer(
                num_channels,
                num_heads,
                chi_irreps,
                fb_rad_features,
                gb_rad_features,
                fb_sph_features,
                gb_sph_features,
                activation,
                residual_mlp_1,
                residual_mlp_2,
                normalization,
                avg_num_neighbors,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs
            )
            for _ in range(num_layers)
        ])

        self.energy_output = MLP(
            [num_channels, num_channels, 1], activation,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(self,
        edge_vectors: jax.Array,
        distances: jax.Array,
        cutoffs: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        edge_feats = self.radial_embedding(distances)

        # This implementation differs from the original So3krates repo: (1) the coefficents are
        # different (So3krates uses physics convention, while here uses math convention), and
        # (2) the output components follow a m-ordering in So3krates, while it is in Cartesian
        # here. These are equivalent.
        edge_vectors = e3nn.IrrepsArray('1e', edge_vectors)
        edge_sh = e3nn.spherical_harmonics(self.chi_irreps, edge_vectors, True)

        # Initalize node features and spherical harmonic coordinates (SPHCs)
        node_feats = self.node_embedding(node_species)
        if self.sphc_normalization is None:
            chi = e3nn.zeros(self.chi_irreps, (node_species.shape[0],), dtype=edge_vectors.dtype)
        else:
            chi = e3nn.scatter_sum(
                edge_sh * cutoffs[:, None], dst=receivers, output_size=node_species.shape[0]
            ) / self.sphc_normalization

        for layer in self.layers:
            node_feats, chi = layer(
                node_feats=node_feats,
                chi=chi,
                edge_feats=edge_feats,
                edge_sh=edge_sh,
                cutoffs=cutoffs,
                senders=senders,
                receivers=receivers
            )

        # node_feats = nnx.LayerNorm()(node_feats)
        node_energies = self.energy_output(node_feats).squeeze(axis=-1)

        return node_energies


class So3kratesLayer(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        num_heads: int,
        chi_irreps: e3nn.Irreps,
        fb_rad_features: list[int],
        gb_rad_features: list[int],
        fb_sph_features: list[int],
        gb_sph_features: list[int],
        activation: str = 'silu',
        residual_mlp_1: bool = False,
        residual_mlp_2: bool = False,
        normalization: bool = False,
        avg_num_neighbors: float = 1.0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.normalization = normalization
        self.residual_mlp_1 = residual_mlp_1
        self.residual_mlp_2 = residual_mlp_2

        if normalization:
            self.ln1 = nnx.LayerNorm(
                num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            self.ln2 = nnx.LayerNorm(
                num_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

        self.feature_block = FeatureBlock(
            num_heads,
            rad_features=fb_rad_features,
            sph_features=fb_sph_features,
            activation=activation,
            avg_num_neighbors=avg_num_neighbors,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.geometric_block = GeometricBlock(
            chi_irreps,
            rad_features=gb_rad_features,
            sph_features=gb_sph_features,
            activation=activation,
            avg_num_neighbors=avg_num_neighbors,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.interaction_block = InteractionBlock(
            num_channels + chi_irreps.num_irreps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        if residual_mlp_1:
            self.mlp1 = ResidualMLP(
                num_channels, activation=activation, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

        if residual_mlp_2:
            self.mlp2 = ResidualMLP(
                num_channels, activation=activation, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray,
        edge_feats: jax.Array,
        edge_sh: e3nn.IrrepsArray,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        chi_ij = chi[senders] - chi[receivers]
        chi_scalar = e3nn.norm(chi_ij, squared=True, per_irrep=True).array

        # first block
        node_feats_pre = self.ln1(node_feats) if self.normalization else node_feats

        diff_node_feats = self.feature_block(
            node_feats=node_feats_pre,
            edge_feats=edge_feats,
            chi_scalar=chi_scalar,
            cutoffs=cutoffs,
            senders=senders,
            receivers=receivers
        )

        diff_chi = self.geometric_block(
            edge_sh=edge_sh,
            node_feats=node_feats_pre,
            edge_feats=edge_feats,
            chi_scalar=chi_scalar,
            cutoffs=cutoffs,
            senders=senders,
            receivers=receivers
        )

        node_feats = node_feats + diff_node_feats
        chi = chi + diff_chi

        # second block
        if self.residual_mlp_1:
            node_feats = self.mlp1(node_feats)

        node_feats_pre = self.ln2(node_feats) if self.normalization else node_feats

        diff_node_feats, diff_chi = self.interaction_block(node_feats_pre, chi)

        node_feats = node_feats + diff_node_feats
        chi = chi + diff_chi

        if self.residual_mlp_2:
            node_feats = self.mlp2(node_feats)

        return node_feats, chi
