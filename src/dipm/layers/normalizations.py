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

import jax
import jax.numpy as jnp


def rms_norm(vec, eps):
    dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + eps)
    dist = jnp.clip(dist, a_min=eps)
    dist = jnp.sqrt(jnp.mean(dist**2, axis=-1))
    return vec / jax.nn.relu(dist).reshape(-1, 1, 1)


def max_min_norm(vec, eps):
    dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + eps)
    direct = vec / jnp.clip(dist, a_min=eps)
    max_val = jnp.max(dist, axis=-1, keepdims=True)
    min_val = jnp.min(dist, axis=-1, keepdims=True)
    delta = max_val - min_val
    delta = delta + eps
    dist = (dist - min_val) / delta
    return jax.nn.relu(dist) * direct


# --- Normalization options ---


class VecNormType(Enum):
    """Options for the VecLayerNorm of the ViSNet model."""

    RMS = "rms"
    MAX_MIN = "max_min"
    NONE = "none"


def get_veclayernorm_fn(norm_type: VecNormType | str, eps: float = 1e-12):
    norm_type_map = {
        VecNormType.RMS: rms_norm,
        VecNormType.MAX_MIN: max_min_norm,
        VecNormType.NONE: lambda x, _: x,
    }
    assert set(VecNormType) == set(norm_type_map.keys())
    norm_fn = norm_type_map[VecNormType(norm_type)]

    def closure(vec):
        if vec.shape[1] != 3 and vec.shape[1] != 8:
            raise ValueError("VecLayerNorm only supports 3 or 8 channels")

        if vec.shape[1] == 3:
            vec = norm_fn(vec, eps)
        elif vec.shape[1] == 8:
            vec1, vec2 = jnp.split(vec, indices_or_sections=[3], axis=1)
            vec1 = norm_fn(vec1, eps)
            vec2 = norm_fn(vec2, eps)
            vec = jnp.concatenate([vec1, vec2], axis=1)

        return vec  # We have removed VecNorm trainability

    return closure
