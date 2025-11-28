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


def rms_norm(vec, eps):
    dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + eps)
    dist = jnp.clip(dist, min=eps)
    dist = jnp.sqrt(jnp.mean(dist**2, axis=-1))
    return vec / jax.nn.relu(dist).reshape(-1, 1, 1)


def max_min_norm(vec, eps):
    dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + eps)
    direct = vec / jnp.clip(dist, min=eps)
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
