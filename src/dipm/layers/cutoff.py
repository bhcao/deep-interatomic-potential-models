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

import jax.numpy as jnp
from flax import nnx


class CosineCutoff(nnx.Module):
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, distances: jnp.ndarray) -> jnp.ndarray:
        cutoffs = 0.5 * (jnp.cos(distances * jnp.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).astype(jnp.float32)
        return cutoffs


class PolynomialCutoff(nnx.Module):
    def __init__(self, cutoff: float, exponent: int = 6):
        self.cutoff = cutoff
        self.exponent = exponent

    def __call__(self, distances: jnp.ndarray) -> jnp.ndarray:
        envelope = (
            1.0
            - ((self.exponent + 1.0) * (self.exponent + 2.0) / 2.0) * jnp.pow(distances / self.cutoff, self.exponent)
            + self.exponent * (self.exponent + 2.0) * jnp.pow(distances / self.cutoff, self.exponent + 1)
            - (self.exponent * (self.exponent + 1.0) / 2.0) * jnp.pow(distances / self.cutoff, self.exponent + 2)
        )
        return envelope * (distances < self.cutoff)
