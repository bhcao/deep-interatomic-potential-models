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

from dipm.layers import Activation
from dipm.typing import PositiveInt


class LiTENConfig(BaseModel):
    """The configuration / hyperparameters of the LiTEN model.

    Attributes:
        num_layers: Number of LiTEN layers. Default is 2.
        num_channels: The number of channels. Default is 256.
        num_heads: Number of heads in the attention block. Default is 8.
        num_rbf: Number of basis functions used in the embedding block. Default is 32.
        trainable_rbf: Whether to add learnable weights to each of the radial embedding
                       basis functions. Default is ``False``.
        activation: Activation function for the output block. Options are "silu"
                    (default), "ssp" (which is shifted softplus), "tanh" and "sigmoid".
        vecnorm_type: The type of the vector norm. The options are "none" (default),
                      "max_min", and "rms".
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default)
                         or the string ``"average"``, then the average atomic energies
                         stored in the dataset info are used. It can also be set to the
                         string ``"zero"`` which means not to use any atomic energies
                         in the model. Lastly, one can also pass an atomic energies
                         dictionary via this parameter different from the one in the
                         dataset info, that is used.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If ``None`` (default), infer the value from the atomic energies
                     map in the dataset info.
    """

    num_layers: PositiveInt = 2
    num_channels: PositiveInt = 256
    num_heads: PositiveInt = 8
    num_rbf: PositiveInt = 32
    trainable_rbf: bool = False
    activation: Activation = Activation.SILU
    atomic_energies: str | dict[int, float] | None = None
    num_species: PositiveInt | None = None
