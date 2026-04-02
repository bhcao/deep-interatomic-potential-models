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

from enum import Enum
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.typing import Dtype
from flax.nnx.nn import initializers, dtypes
import e3nn_jax as e3nn


class ExpNormalBasis(nnx.Module):
    """Original ExpNormalSmearing from Visnet without cutoff function."""
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
        **_kwargs, # to capture rbf_width
    ):
        self.cutoff = cutoff
        self.alpha = 5.0 / cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        if trainable:
            means, betas = self._initial_params(cutoff, num_rbf, param_dtype)
            means_key = rngs.params()
            self.means = nnx.Param(
                initializers.constant(means)(means_key, (num_rbf,), param_dtype)
            )
            betas_key = rngs.params()
            self.betas = nnx.Param(
                initializers.constant(betas)(betas_key, (num_rbf,), param_dtype)
            )

        self.dtype = dtype

    @cache
    @staticmethod
    def _initial_params(cutoff, num_rbf, dtype):
        start_value = jnp.exp(-cutoff)
        means = jnp.linspace(start_value, 1, num_rbf, dtype=dtype)
        betas = jnp.full(
            (num_rbf,), (2 / num_rbf * (1 - start_value)) ** -2, dtype=dtype
        )
        return means, betas

    def __call__(self, dist: jax.Array) -> jax.Array:
        if self.trainable:
            dist, betas, means = dtypes.promote_dtype(
                (dist, self.betas.value, self.means.value), dtype=self.dtype
            )
        else:
            dist, = dtypes.promote_dtype((dist,), dtype=self.dtype)
            means, betas = self._initial_params(self.cutoff, self.num_rbf, self.dtype)

        dist = dist[..., None]
        return jnp.exp(
            (-1 * betas) * (jnp.exp(self.alpha * (-dist)) - means) ** 2
        )


class GaussianBasis(nnx.Module):
    """
    Original GaussianSmearing from Visnet without cutoff function.
    It's also used in So3krates named RBF.
    """
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
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.rbf_width = rbf_width
        self.trainable = trainable

        if trainable:
            offset, coeff = self._initial_params(cutoff, num_rbf, rbf_width, param_dtype)
            offset_key = rngs.params()
            self.offset = nnx.Param(
                initializers.constant(offset)(offset_key, (num_rbf,), param_dtype)
            )
            coeff_key = rngs.params()
            self.coeff = nnx.Param(
                initializers.constant(coeff)(coeff_key, (), param_dtype)
            )

        self.dtype = dtype

    @cache
    @staticmethod
    def _initial_params(cutoff, num_rbf, rbf_width, dtype):
        offset = jnp.linspace(0, cutoff, num_rbf, dtype=dtype)
        coeff = -0.5 / (rbf_width * (offset[1] - offset[0])) ** 2
        return offset, coeff

    def __call__(self, dist: jax.Array) -> jax.Array:
        if self.trainable:
            dist, offset, coeff = dtypes.promote_dtype(
                (dist, self.offset.value, self.coeff.value), dtype=self.dtype
            )
        else:
            dist, = dtypes.promote_dtype((dist,), dtype=self.dtype)
            offset, coeff = self._initial_params(
                self.cutoff, self.num_rbf, self.rbf_width, self.dtype
            )

        dist = dist[..., None] - offset
        return jnp.exp(coeff * jnp.square(dist))


class BesselBasis(nnx.Module):
    """Bessel basis used in Mace and Nequip. This is not the same named function in So3krates."""

    # **_kwargs is to capture dtype, param_dtype etc.
    def __init__(self, cutoff: float, num_rbf: int, **_kwargs):
        self.num_rbf = num_rbf
        self.cutoff = cutoff

    def __call__(self, dist: jax.Array) -> jax.Array:
        return e3nn.bessel(dist, self.num_rbf, self.cutoff)


def log_binomial(n: int) -> jax.Array:
    """
    Returns: jax.Array of shape (n+1,)
    [log C(n, 0), ..., log C(n, n)]
    """
    out = []
    for k in range(n + 1):
        n_factorial = np.sum(np.log(np.arange(1, n + 1)))
        k_factorial = np.sum(np.log(np.arange(1, k + 1)))
        n_k_factorial = np.sum(np.log(np.arange(1, n - k + 1)))
        out.append(n_factorial - k_factorial - n_k_factorial)
    return jnp.stack(out)


class BernsteinBasis(nnx.Module):
    """Bernstein polynomial basis from So3krates."""
    def __init__(self, cutoff: float, num_rbf: int, gamma: float = 0.9448630629184640, **_kwargs):
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.gamma = gamma

    def __call__(self, dist: jax.Array) -> jax.Array:
        b = log_binomial(self.num_rbf - 1)
        k = jnp.arange(self.num_rbf)
        k_rev = k[::-1]

        scaled_dist = -self.gamma * dist[..., None]
        k_x = k * scaled_dist
        kk_x = k_rev * jnp.log(1e-8 - jnp.expm1(scaled_dist))
        return jnp.exp(b + k_x + kk_x)


class PhysNetBasis(nnx.Module):
    """Expand distances in the basis used in PhysNet (see https://arxiv.org/abs/1902.08408)"""
    def __init__(self, cutoff: float, num_rbf: int, **_kwargs):
        self.num_rbf = num_rbf
        self.cutoff = cutoff

    def __call__(self, dist: jax.Array) -> jax.Array:
        exp_dist = jnp.exp(-dist)[..., None]
        exp_cutoff = jnp.exp(-self.cutoff)

        offset = jnp.linspace(exp_cutoff, 1, self.num_rbf)
        coeff = self.num_rbf / 2 / (1 - exp_cutoff)
        return jnp.exp(-(coeff * (exp_dist - offset)) ** 2)


class FourierBasis(nnx.Module):
    """
    Expand distances in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf).
    It's also called Bessel basis in So3krates. Since we already have the BesselBasis, we have to
    use another name.
    """
    def __init__(self, cutoff: float, num_rbf: int, **_kwargs):
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.offset = jnp.arange(0, self.num_rbf, 1)

    def __call__(self, dist: jax.Array) -> jax.Array:
        dist = dist[..., None]
        # In So3krates, safe_mask is used to avoid divide by zero. Here, we use a small epsilon.
        return jnp.sin(jnp.pi / self.cutoff * self.offset * dist) / (dist + 1e-8)


# --- Radial options ---


class RadialBasis(Enum):
    """Radial basis option(s)."""

    GAUSS = "gauss"
    EXPNORM = "expnorm"
    BESSEL = "bessel"
    BERNSTEIN = "bernstein"
    PHYS = "phys"
    FOURIER = "fourier"


def get_radial_basis_cls(basis: RadialBasis | str) -> type[nnx.Module]:
    """Parse `RadialBasis` parameter among available options."""

    radial_basis_map = {
        RadialBasis.GAUSS: GaussianBasis,
        RadialBasis.EXPNORM: ExpNormalBasis,
        RadialBasis.BESSEL: BesselBasis,
        RadialBasis.BERNSTEIN: BernsteinBasis,
        RadialBasis.PHYS: PhysNetBasis,
        RadialBasis.FOURIER: FourierBasis,
    }
    assert set(RadialBasis) == set(radial_basis_map.keys())
    return radial_basis_map[RadialBasis(basis)]
