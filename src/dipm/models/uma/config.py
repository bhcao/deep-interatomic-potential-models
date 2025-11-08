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

from pydantic import BaseModel

from dipm.layers.escn import LayerNormType
from dipm.typing import PositiveInt, NonNegativeInt, DtypeEnum
from dipm.models.uma.blocks import FeedForwardType, ActivationType


class UMAConfig(BaseModel):
    """The configuration / hyperparameters of the UMA model.

    Attributes:
        num_layers: Number of UMA layers. Default is 12.
        lmax: Maximum degree of the spherical harmonics (1 to 10).
        mmax: Maximum order of the spherical harmonics (0 to lmax).
        sphere_channels: Number of spherical channels. Default is 128.
        edge_channels: Number of channels for the edge invariant features. Default is 128.
        hidden_channels: Number of hidden channels in the UMA layer. Default is 128.
        num_rbf: Number of basis functions used in the embedding block. Default is 600.
        grid_resolution: Resolution of SO3Grid used in Activation. Examples are 18, 16, 14, None
                         (default, decided automatically).
        norm_type: Type of normalization layer. Options are "layer_norm", "layer_norm_sh" (default)
                   and "rms_norm_sh".
        act_type: Type of activation function. Options are "gate" (default) and "s2_sep".
        ff_type: Type of feed-forward function. Options are "spectral" (default) and "grid".
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default) or the
                         string ``"average"``, then the average atomic energies stored in the
                         dataset info are used. It can also be set to the string ``"zero"`` which
                         means not to use any atomic energies in the model. Lastly, one can also
                         pass an atomic energies dictionary via this parameter different from the
                         one in the dataset info, that is used.
        dataset_list: List of different datasets used in training. Every dataset will have a
                      different embedding. `None` means no dataset embedding.
        num_species: The number of elements (atomic species descriptors) allowed. If ``None``
                     (default), infer the value from the atomic energies map in the dataset info.
        num_experts: Number of experts in the MoLE block. Default is 8.
        mole_dropout: Dropout rate for MoLE router. Default is 0.0.
        use_composition_embedding: Whether to use composition embedding in MoLE router.
        param_dtype: The data type of model parameters. Default is ``jnp.float32``.
        force_head: Whether to predict forces with forces head. Default is ``False``.
    """

    num_layers: PositiveInt = 12
    lmax: PositiveInt = 6
    mmax: NonNegativeInt = 2
    sphere_channels: PositiveInt = 128
    edge_channels: PositiveInt = 128
    hidden_channels: PositiveInt = 128
    num_rbf: PositiveInt = 600
    grid_resolution: PositiveInt = None
    norm_type: LayerNormType = LayerNormType.LAYER_NORM_SH
    act_type: ActivationType = ActivationType.GATE
    ff_type: FeedForwardType = FeedForwardType.GRID
    atomic_energies: str | dict[int, float] | None = None
    num_species: PositiveInt | None = None
    dataset_list: list[str] | None = ["oc20", "omol", "omat", "odac", "omc"]
    num_experts: int = 8
    mole_dropout: float = 0.0
    use_composition_embedding: bool = False
    param_dtype: DtypeEnum = DtypeEnum.F32
    force_head: bool = False
