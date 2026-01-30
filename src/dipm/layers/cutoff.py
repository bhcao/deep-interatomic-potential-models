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

import jax
import jax.numpy as jnp
from flax import nnx
import e3nn_jax as e3nn


class SoftCutoff(nnx.Module):
    """Soft envelope radial envelope function."""
    def __init__(
        self, cutoff: float, arg_multiplicator: float = 2.0, value_at_origin: float = 1.2, **_kwargs
    ):
        self.cutoff = cutoff
        self.arg_multiplicator = arg_multiplicator
        self.value_at_origin = value_at_origin

    def __call__(self, length):
        return e3nn.soft_envelope(
            length,
            self.cutoff,
            arg_multiplicator=self.arg_multiplicator,
            value_at_origin=self.value_at_origin,
        )


class PolynomialCutoff(nnx.Module):
    """
    From the MACE torch version, referenced to:
    Klicpera, J.; Groß, J.; Günnemann, S.
    Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """
    def __init__(self, cutoff: float, exponent: int = 5, **_kwargs):
        self.cutoff = cutoff
        self.p = exponent
        self.a = - (exponent + 1.0) * (exponent + 2.0) / 2.0
        self.b = exponent * (exponent + 2.0)
        self.c = - exponent * (exponent + 1.0) / 2

    def __call__(self, length: jax.Array):
        x_norm = length / self.cutoff
        envelope = 1.0 + jnp.pow(x_norm, self.p) * (
            self.a + x_norm * (self.b + x_norm * self.c)
        )
        return envelope * (length < self.cutoff)


class CosineCutoff(nnx.Module):
    """Behler-style cosine cutoff function."""
    def __init__(self, cutoff: float, **_kwargs):
        self.cutoff = cutoff

    def __call__(self, length: jax.Array) -> jax.Array:
        cutoffs = 0.5 * (jnp.cos(length * jnp.pi / self.cutoff) + 1.0)
        return cutoffs * (length < self.cutoff)


class PhysCutoff(nnx.Module):
    """Cutoff function used in PhysNet."""
    def __init__(self, cutoff: float, **_kwargs):
        self.cutoff = cutoff

    def __call__(self, length: jax.Array) -> jax.Array:
        x_norm = length / self.cutoff
        cutoffs = 1 - 6 * x_norm ** 5 + 15 * x_norm ** 4 - 10 * x_norm ** 3
        return cutoffs * (length < self.cutoff)


class ExponentialCutoff(nnx.Module):
    """Exponential cutoff function used in SpookyNet."""
    def __init__(self, cutoff: float, **_kwargs):
        self.cutoff = cutoff

    def __call__(self, length: jax.Array) -> jax.Array:
        cutoffs = jnp.exp(-length ** 2 / ((self.cutoff - length) * (self.cutoff + length)))
        return cutoffs * (length < self.cutoff)

# --- Options ---


class CutoffFunction(Enum):
    """Radial envelope / cutoff function options."""

    POLYNOMIAL = "polynomial"
    SOFT = "soft"
    COSINE = "cosine"
    PHYS = "phys"
    EXPONENTIAL = "exponential"


def get_cutoff_cls(cutoff: CutoffFunction | str) -> type[nnx.Module]:
    """Parse `RadialEnvelope` parameter among available options."""

    cutoff_function_map = {
        CutoffFunction.POLYNOMIAL: PolynomialCutoff,
        CutoffFunction.SOFT: SoftCutoff,
        CutoffFunction.COSINE: CosineCutoff,
        CutoffFunction.PHYS: PhysCutoff,
        CutoffFunction.EXPONENTIAL: ExponentialCutoff,
    }
    assert set(CutoffFunction) == set(cutoff_function_map.keys())
    return cutoff_function_map[CutoffFunction(cutoff)]
