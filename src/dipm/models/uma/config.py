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

from dipm.layers.escn import LayerNormType
from dipm.typing import PositiveInt, NonNegativeInt
from dipm.models.uma.blocks import FeedForwardType, ActivationType
from dipm.models.force_model import ForceModelConfig


class UMAConfig(ForceModelConfig):
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
        num_experts: Number of experts in the MoLE block. Default is 8.
        mole_dropout: Dropout rate for MoLE router. Default is 0.0.
        use_composition_embedding: Whether to use composition embedding in MoLE router.
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
    num_experts: int = 8
    mole_dropout: float = 0.0
    use_composition_embedding: bool = False
