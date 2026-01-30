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

from dipm.layers import (
    CutoffFunction,
    RadialBasis,
    Activation
)
from dipm.typing import PositiveInt
from dipm.models.force_model import ForceModelConfig


class So3kratesConfig(ForceModelConfig):
    """Hyperparameters for the So3krates model.

    Attributes:
        num_layers: Number of So3krates layers. Default is 3.
        num_channels: The number of channels. Default is 128.
        num_heads: Number of heads in the attention block. Default is 4.
        num_rbf: Number of basis functions used in the embedding block. Default is 32.
        activation: Activation function for the output block. Options are "silu"
                    (default), "ssp" (which is shifted softplus), "tanh", "sigmoid", and
                    "swish".
        radial_cutoff_fn: The type of the cutoff / radial envelope function.
        radial_basis_fn: The type of the radial basis function.
        chi_irreps: The irreps of the spherical harmonic coorindates (SPHCs).
        sphc_normalization: Normalization constant for initializing spherical harmonic
                            coordinates (SPHCs). If set to ``None``, SPHCs are initialized
                            to zero.
        residual_mlp_1: Whether to apply a residual MLP after the first (feature + 
                        geometric) update block inside each So3krates layer.
        residual_mlp_2: Whether to apply a residual MLP after the interaction block inside
                        each So3krates layer.
        normalization: Whether to apply LayerNorm to scalar node features before major
                       update blocks inside each So3krates layer.
        zbl_repulsion: Whether to include an explicit Ziegler-Biersac-Littmark (ZBL)
                       short-range nuclear repulsion term in the predicted energies.
        zbl_repulsion_shift: Constant energy shift subtracted from the ZBL repulsion
                             contribution.
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default)
                         or the string ``"average"``, then the average atomic energies
                         stored in the dataset info are used. It can also be set to the
                         string ``"zero"`` which means not to use any atomic energies
                         in the model. Lastly, one can also pass an atomic energies
                         dictionary via this parameter different from the one in the
                         dataset info, that is used.
        avg_num_neighbors: The mean number of neighbors for atoms. If ``None``
                           (default), use the value from the dataset info.
                           It is used to rescale messages by this value.
    """

    num_layers: PositiveInt = 3
    num_channels: PositiveInt = 128
    num_heads: PositiveInt = 4
    num_rbf: PositiveInt = 32
    activation: Activation = Activation.SILU
    radial_cutoff_fn: CutoffFunction = CutoffFunction.PHYS
    radial_basis_fn: RadialBasis = RadialBasis.BERNSTEIN
    chi_irreps: str = "1e + 2e + 3e + 4e"
    sphc_normalization: float | None = None
    residual_mlp_1: bool = True
    residual_mlp_2: bool = False
    normalization: bool = True
    zbl_repulsion: bool = True
    zbl_repulsion_shift: float = 0.0
    atomic_energies: str | dict[int, float] | None = None
    avg_num_neighbors: float | None = None
