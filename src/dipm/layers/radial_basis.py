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

from enum import Enum
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers, dtypes
import e3nn_jax as e3nn

from dipm.layers.radial_embeddings import CosineCutoff


class ExpNormalSmearing(nnx.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.alpha = 5.0 / cutoff
        self.trainable = trainable

        means, betas = self._initial_params(cutoff, num_rbf)
        if trainable:
            means_key = rngs.params()
            self.means = nnx.Param(
                initializers.constant(means)(means_key, (num_rbf,), param_dtype)
            )
            betas_key = rngs.params()
            self.betas = nnx.Param(
                initializers.constant(betas)(betas_key, (num_rbf,), param_dtype)
            )
        else:
            self.means = nnx.Cache(means)
            self.betas = nnx.Cache(betas)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.dtype = dtype

    def _initial_params(self, cutoff, num_rbf):
        start_value = jnp.exp(-cutoff)
        means = jnp.linspace(start_value, 1, num_rbf)
        betas = jnp.full((num_rbf,), (2 / num_rbf * (1 - start_value)) ** -2)
        return means, betas

    def __call__(self, dist: jax.Array) -> jax.Array:
        dist, betas, means = dtypes.promote_dtype(
            (dist, self.betas.value, self.means.value), dtype=self.dtype
        )

        dist = dist[..., None]
        cutoffs = self.cutoff_fn(dist)
        return cutoffs * jnp.exp(
            (-1 * betas) * (jnp.exp(self.alpha * (-dist)) - means) ** 2
        )


class GaussianSmearing(nnx.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        rbf_width: float = 1.0,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.trainable = trainable

        offset, coeff = self._initial_params(cutoff, num_rbf, rbf_width)
        if trainable:
            offset_key = rngs.params()
            self.offset = nnx.Param(
                initializers.constant(offset)(offset_key, (num_rbf,), param_dtype)
            )
            coeff_key = rngs.params()
            self.coeff = nnx.Param(
                initializers.constant(coeff)(coeff_key, (), param_dtype)
            )
        else:
            self.offset = nnx.Cache(offset)
            self.coeff = nnx.Cache(coeff)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.dtype = dtype

    def _initial_params(self, cutoff, num_rbf, rbf_width):
        offset = jnp.linspace(0, cutoff, num_rbf)
        coeff = -0.5 / (rbf_width * (offset[1] - offset[0])) ** 2
        return offset, coeff

    def __call__(self, dist: jax.Array) -> jax.Array:
        dist, offset, coeff = dtypes.promote_dtype(
            (dist, self.offset.value, self.coeff.value), dtype=self.dtype
        )

        dist = dist[..., None] - offset
        cutoffs = self.cutoff_fn(dist)
        return cutoffs * jnp.exp(coeff * jnp.square(dist))


def bessel_basis(length: jax.Array, max_length: float, number: int) -> jax.Array:
    """Returns the Bessel function with given length, max. length, and number."""
    return e3nn.bessel(length, number, max_length)


# --- Radial options ---


class RadialBasis(Enum):
    """Radial basis option(s). For the moment, only `BESSEL = "bessel"` exists."""

    BESSEL = "bessel"


class VisnetRBF(Enum):
    """Options for the radial basis functions used by ViSNet."""

    GAUSS = "gauss"
    EXPNORM = "expnorm"


def get_radial_basis_fn(basis: RadialBasis | str) -> Callable:
    """Parse `RadialBasis` parameter among available options.

    See :class:`~dipm.models.options.RadialBasis`.
    """
    radial_basis_map = {
        RadialBasis.BESSEL: bessel_basis,
    }
    return radial_basis_map[RadialBasis(basis)]


def get_rbf_cls(rbf_type: VisnetRBF | str) -> Callable:
    '''Mapping of RBF class names to their Flax classes'''
    rbf_class_mapping = {
        VisnetRBF.GAUSS: GaussianSmearing,
        VisnetRBF.EXPNORM: ExpNormalSmearing,
    }
    return rbf_class_mapping[VisnetRBF(rbf_type)]
