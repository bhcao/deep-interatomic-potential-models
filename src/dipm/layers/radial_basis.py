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

from enum import Enum
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers
import e3nn_jax as e3nn

from dipm.layers.cutoff import CosineCutoff


class ExpNormalSmearing(nnx.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        *,
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
            self.means = means
            self.betas = betas
        self.cutoff_fn = CosineCutoff(cutoff)

    def _initial_params(self, cutoff, num_rbf):
        start_value = jnp.exp(-cutoff)
        means = jnp.linspace(start_value, 1, num_rbf)
        betas = jnp.full((num_rbf,), (2 / num_rbf * (1 - start_value)) ** -2)
        return means, betas

    def __call__(self, dist: jnp.ndarray) -> jnp.ndarray:
        betas = self.betas.value if self.trainable else self.betas
        means = self.means.value if self.trainable else self.means

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
        *,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.trainable = trainable

        offset, coeff = self._initial_params(cutoff, num_rbf)
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
            self.offset = offset
            self.coeff = coeff
        self.cutoff_fn = CosineCutoff(cutoff)

    def _initial_params(self, cutoff, num_rbf):
        offset = jnp.linspace(0, cutoff, num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def __call__(self, dist: jnp.ndarray) -> jnp.ndarray:
        offset = self.offset.value if self.trainable else self.offset
        coeff = self.coeff.value if self.trainable else self.coeff

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
