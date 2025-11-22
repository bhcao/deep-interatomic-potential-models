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

from dipm.layers import Activation
from dipm.typing import PositiveInt
from dipm.models.force_model import ForceModelConfig


class LiTENConfig(ForceModelConfig):
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
    """

    num_layers: PositiveInt = 2
    num_channels: PositiveInt = 256
    num_heads: PositiveInt = 8
    num_rbf: PositiveInt = 32
    trainable_rbf: bool = False
    activation: Activation = Activation.SILU
    atomic_energies: str | dict[int, float] | None = None
