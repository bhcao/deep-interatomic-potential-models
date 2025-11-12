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
from flax import nnx
from flax.nnx.nn import initializers, dtypes
from flax.typing import Dtype

from dipm.layers.escn.utils import expand_index


class EquivariantLayerNormArray(nnx.Module):
    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        if affine:
            self.affine_weight = nnx.Param(
                initializers.ones(rngs.params(), (lmax + 1, num_channels), param_dtype)
            )
            self.affine_bias = nnx.Param(
                initializers.zeros(rngs.params(), num_channels, param_dtype)
            )
        else:
            self.affine_weight = None
            self.affine_bias = None

        self.lmax = lmax
        self.eps = eps
        self.affine = affine
        self.normalization = normalization
        self.dtype = dtype

    def __call__(self, node_input: jax.Array):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        affine_weight = self.affine_weight.value if self.affine_weight is not None else None
        affine_bias = self.affine_bias.value if self.affine_bias is not None else None

        node_input, affine_weight, affine_bias = dtypes.promote_dtype(
            (node_input, affine_weight, affine_bias), dtype=self.dtype
        )

        out = []

        for lval in range(self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1

            feature = node_input[:, start_idx : start_idx+length]

            # For scalars, first compute and subtract the mean
            if lval == 0:
                feature_mean = jnp.mean(feature, axis=2, keepdims=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = jnp.sum(
                    jnp.pow(feature, 2), axis=1, keepdims=True
                ) # [N, 1, C]
            elif self.normalization == "component":
                feature_norm = jnp.mean(
                    jnp.pow(feature, 2), axis=1, keepdims=True
                ) # [N, 1, C]

            feature_norm = jnp.mean(
                feature_norm, axis=2, keepdims=True
            )  # [N, 1, 1]
            feature_norm = jnp.pow(feature_norm + self.eps, -0.5)

            if self.affine:
                weight = affine_weight[None, lval:lval+1]  # [1, 1, C]
                feature_norm = feature_norm * weight  # [N, 1, C]

            feature = feature * feature_norm

            if self.affine and lval == 0:
                bias = affine_bias[None, None]
                feature = feature + bias

            out.append(feature)

        out = jnp.concat(out, axis=1)
        return out


class EquivariantLayerNormArraySphericalHarmonics(nnx.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        std_balance_degrees: bool = True,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        # for L = 0
        self.norm_l0 = nnx.LayerNorm(
            num_channels, epsilon=eps, use_bias=affine, use_scale=affine,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # for L > 0
        if affine:
            self.affine_weight = nnx.Param(
                initializers.ones(rngs.params(), (lmax, num_channels), param_dtype)
            )
        else:
            self.affine_weight = None

        self.normalization = normalization

        if std_balance_degrees:
            balance_degree_weight = jnp.zeros(((lmax + 1) ** 2 - 1, 1), dtype=dtype)
            for lval in range(1, lmax + 1):
                start_idx = lval**2 - 1
                length = 2 * lval + 1
                balance_degree_weight = balance_degree_weight.at[
                    start_idx : (start_idx + length), :
                ].set(1.0 / length)
            self.balance_degree_weight = nnx.Cache(balance_degree_weight / lmax)
        else:
            self.balance_degree_weight = None

        self.lmax = lmax
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.dtype = dtype

    def __call__(self, node_input: jax.Array):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        affine_weight = self.affine_weight.value if self.affine_weight is not None else None

        node_input, affine_weight = dtypes.promote_dtype(
            (node_input, affine_weight), dtype=self.dtype
        )

        out = []

        # for L = 0
        feature = node_input[:, :1]
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input[:, 1:num_m_components]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = jnp.sum(
                    jnp.pow(feature, 2), axis=1, keepdims=True
                )  # [N, 1, C]
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = jnp.pow(feature, 2)
                    feature_norm = jnp.einsum(
                        "nic, ia -> nac",
                        feature_norm,
                        self.balance_degree_weight.value,
                    )  # [N, 1, C]
                else:
                    feature_norm = jnp.mean(
                        jnp.pow(feature, 2), axis=1, keepdims=True
                    )  # [N, 1, C]

            feature_norm = jnp.mean(
                feature_norm, axis=2, keepdims=True
            )  # [N, 1, 1]
            feature_norm = jnp.pow(feature_norm + self.eps, -0.5)

            for lval in range(1, self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                # [N, (2L + 1), C]
                feature = node_input[:, start_idx : start_idx+length]
                if self.affine:
                    weight = affine_weight[None, lval-1:lval] # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        out = jnp.concat(out, axis=1)

        return out


class EquivariantRMSNormArraySphericalHarmonicsV2(nnx.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        centering: bool = True,
        std_balance_degrees: bool = True,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        # for L >= 0
        if affine:
            self.affine_weight = nnx.Param(
                initializers.ones(rngs.params(), (lmax + 1, num_channels), param_dtype)
            )
            if centering:
                self.affine_bias = nnx.Param(
                    initializers.zeros(rngs.params(), num_channels, param_dtype)
                )
            else:
                self.affine_bias = None
        else:
            self.affine_weight = None
            self.affine_bias = None

        if normalization == "norm":
            assert not std_balance_degrees

        self.expand_index = nnx.Cache(expand_index(lmax))

        if std_balance_degrees:
            balance_degree_weight = jnp.zeros(((lmax + 1) ** 2, 1), dtype=dtype)
            for lval in range(lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight = balance_degree_weight.at[
                    start_idx : (start_idx + length), :
                ].set(1.0 / length)
            self.balance_degree_weight = nnx.Cache(balance_degree_weight / (lmax + 1))
        else:
            self.balance_degree_weight = None

        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        self.normalization = normalization
        self.dtype = dtype

    def __call__(self, node_input: jax.Array):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        affine_weight = self.affine_weight.value if self.affine_weight is not None else None
        affine_bias = self.affine_bias.value if self.affine_bias is not None else None

        node_input, affine_weight, affine_bias = dtypes.promote_dtype(
            (node_input, affine_weight, affine_bias), dtype=self.dtype
        )

        feature = node_input

        if self.centering:
            feature_l0 = feature[:, 0:1]
            feature_l0_mean = jnp.mean(feature_l0, axis=2, keepdims=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = jnp.concat(
                (feature_l0, feature[:, 1:feature.shape[1]]), axis=1
            )

        # for L >= 0
        if self.normalization == "norm":
            feature_norm = jnp.sum(
                jnp.pow(feature, 2), axis=1, keepdims=True
            )  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = jnp.pow(feature, 2)  # [N, (L_max + 1)**2, C]
                feature_norm = jnp.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight.value
                )  # [N, 1, C]
            else:
                feature_norm = jnp.mean(
                    jnp.pow(feature, 2), axis=1, keepdims=True
                )  # [N, 1, C]

        feature_norm = jnp.mean(
            feature_norm, axis=2, keepdims=True
        )  # [N, 1, 1]
        feature_norm = jnp.pow(feature_norm + self.eps, -0.5)

        if self.affine:
            # [1, (L_max + 1)**2, C]
            weight = affine_weight[None, self.expand_index.value]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out = out.at[:, 0:1, :].set(
                out[:, 0:1] + affine_bias[None, None]
            )

        return out


# --- Normalization options ---


class LayerNormType(Enum):
    """Options for the LayerNorm of the EquiformerV2 model."""

    LAYER_NORM = "layer_norm"
    LAYER_NORM_SH = "layer_norm_sh"
    RMS_NORM_SH = "rms_norm_sh"


def get_layernorm_layer(
    norm_type: LayerNormType | str,
    lmax: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = True,
    normalization: str = "component",
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    rngs: nnx.Rngs
):
    assert normalization in ["norm", "component"]
    norm_type_map = {
        LayerNormType.LAYER_NORM: EquivariantLayerNormArray,
        LayerNormType.LAYER_NORM_SH: EquivariantLayerNormArraySphericalHarmonics,
        LayerNormType.RMS_NORM_SH: EquivariantRMSNormArraySphericalHarmonicsV2,
    }
    norm_class = norm_type_map[LayerNormType(norm_type)]
    return norm_class(lmax, num_channels, eps, affine, normalization,
                      dtype=dtype, param_dtype=param_dtype, rngs=rngs)
