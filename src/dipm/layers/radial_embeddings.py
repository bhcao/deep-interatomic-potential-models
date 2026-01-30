# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
#
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
        avg_r_min: float | None = None,
    ):
        self.r_max = r_max
        self.avg_r_min = avg_r_min

        def func(lengths):
            basis = basis_functions(lengths)  # [n_edges, num_bessel]
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
                    self.avg_r_min, self.r_max, 1000, dtype=edge_lengths.dtype
                )
                factor = jnp.mean(self.func(samples) ** 2).item() ** -0.5

        embedding = factor * jnp.where(
            (edge_lengths == 0.0)[:, None], 0.0, self.func(edge_lengths)
        )  # [n_edges, num_bessel]

        return e3nn.IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)
