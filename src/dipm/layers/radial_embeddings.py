# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
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

from enum import Enum
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
import e3nn_jax as e3nn

class RadialEmbeddingLayer(nnx.Module):
    """Radial encoding of interatomic distances."""

    def __init__(
        self,
        r_max: float,
        basis_functions: Callable[[jax.Array], jax.Array],
        envelope_function: Callable[[jax.Array], jax.Array],
        num_bessel: int,
        avg_r_min: float | None = None,
    ):
        self.r_max = r_max
        self.avg_r_min = avg_r_min

        def func(lengths):
            basis = basis_functions(
                lengths,
                r_max,
                num_bessel,
            )  # [n_edges, num_bessel]
            cutoff = envelope_function(lengths)  # [n_edges]
            return basis * cutoff[:, None]  # [n_edges, num_bessel]

        self.func = func

    def __call__(
        self,
        edge_lengths: jax.Array,  # [n_edges]
    ) -> e3nn.IrrepsArray:  # [n_edges, num_bessel]
        with jax.ensure_compile_time_eval():
            if self.avg_r_min is None:
                factor = 1.0
            else:
                samples = jnp.linspace(
                    self.avg_r_min, self.r_max, 1000, dtype=jnp.float32
                )
                factor = jnp.mean(self.func(samples) ** 2).item() ** -0.5

        embedding = factor * jnp.where(
            (edge_lengths == 0.0)[:, None], 0.0, self.func(edge_lengths)
        )  # [n_edges, num_bessel]

        return e3nn.IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)


class SoftEnvelope(nnx.Module):
    """Soft envelope radial envelope function."""
    def __init__(
        self, cutoff: float, arg_multiplicator: float = 2.0, value_at_origin: float = 1.2
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
    def __init__(self, cutoff: float, exponent: int = 5):
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
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, length: jax.Array) -> jax.Array:
        cutoffs = 0.5 * (jnp.cos(length * jnp.pi / self.cutoff) + 1.0)
        return cutoffs * (length < self.cutoff)


# --- Radial options ---


class RadialEnvelope(Enum):
    """Radial envelope options. For the moment,
    `POLYNOMIAL = "polynomial_envelope"` and `SOFT = "soft_envelope"` exist.
    """

    POLYNOMIAL = "polynomial_envelope"
    SOFT = "soft_envelope"


def get_radial_envelope_cls(envelope: RadialEnvelope | str) -> Callable:
    """Parse `RadialEnvelope` parameter among available options.

    See :class:`~dipm.models.options.RadialEnvelope`."""
    radial_envelope_map = {
        RadialEnvelope.POLYNOMIAL: PolynomialCutoff,
        RadialEnvelope.SOFT: SoftEnvelope,
    }
    assert set(RadialEnvelope) == set(radial_envelope_map.keys())
    return radial_envelope_map[RadialEnvelope(envelope)]
