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


class EquiformerV2Config(BaseModel):
    """The configuration / hyperparameters of the EquiformerV2 model.

    Attributes:
        num_layers: Number of EquiformerV2 layers. Default is 12.
        lmax: Maximum degree of the spherical harmonics (1 to 10).
        mmax: Maximum order of the spherical harmonics (0 to lmax).
        sphere_channels: Number of spherical channels. Default is 128.
        num_edge_channels: Number of channels for the edge invariant features. Default is 128.
        atom_edge_embedding: Whether to use / share atomic embedding along with relative distance.
                             Options are "none" (default), "isolated" and "shared".
        num_rbf: Number of basis functions used in the embedding block. Default is 600.
        attn_hidden_channels: Number of hidden channels used during SO(2) graph attention. Use 64
                              or 96 (not necessarily).
        num_heads: Number of heads in the attention block. Default is 8.
        attn_alpha_channels: Number of channels for alpha vector in each attention head.
        attn_value_channels: Number of channels for value vector in each attention head.
        ffn_hidden_channels: Number of hidden channels used during feedforward network.
        norm_type: Type of normalization layer. Options are "layer_norm", "layer_norm_sh" (default)
                   and "rms_norm_sh".
        grid_resolution: Resolution of SO3Grid used in Activation. Examples are 18, 16, 14, None
                         (default, decided automatically).
        use_m_share_rad: Whether all m components within a type-L vector of one channel share
                         radial function weights.
        use_attn_renorm: Whether to re-normalize attention weights.
        use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation.
        use_grid_mlp: If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False.
        alpha_drop: Dropout rate for attention weights. Use 0.0 or 0.1 (default).
        drop_path_rate: Grpah drop path rate. Use 0.0 or 0.05 (default).
        avg_num_nodes: The mean number of atoms per graph. If `None`, use the value from the
                       dataset info. Default is value from IS2RE (100k).
        avg_num_neighbors: The mean number of neighbors for atoms. If `None`, use the value
                           from the dataset info. Default is value from IS2RE (100k). It is
                           used to rescale messages by this value.
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default) or the
                         string ``"average"``, then the average atomic energies stored in the
                         dataset info are used. It can also be set to the string ``"zero"`` which
                         means not to use any atomic energies in the model. Lastly, one can also
                         pass an atomic energies dictionary via this parameter different from the
                         one in the dataset info, that is used.
        num_species: The number of elements (atomic species descriptors) allowed. If ``None``
                     (default), infer the value from the atomic energies map in the dataset info.
        param_dtype: The data type of model parameters. Default is ``jnp.float32``.
        force_head: Whether to predict forces with forces head. Default is ``False``.
    """

    num_layers: PositiveInt = 12
    lmax: PositiveInt = 6
    mmax: NonNegativeInt = 2
    sphere_channels: PositiveInt = 128
    num_edge_channels: PositiveInt = 128
    atom_edge_embedding: str = 'isolated'
    num_rbf: PositiveInt = 600
    attn_hidden_channels: PositiveInt = 64
    num_heads: PositiveInt = 8
    attn_alpha_channels: PositiveInt = 64
    attn_value_channels: PositiveInt = 16
    ffn_hidden_channels: PositiveInt = 128
    norm_type: LayerNormType = LayerNormType.LAYER_NORM_SH
    grid_resolution: PositiveInt = None
    use_m_share_rad: bool = False
    use_attn_renorm: bool = True
    use_gate_act: bool = False
    use_grid_mlp: bool = True
    use_sep_s2_act: bool = True
    alpha_drop: float = 0.1
    drop_path_rate: float = 0.05
    avg_num_neighbors: float | None = 23.395238876342773
    avg_num_nodes: float | None = 77.81317
    atomic_energies: str | dict[int, float] | None = None
    num_species: PositiveInt | None = None
    param_dtype: DtypeEnum = DtypeEnum.F32
    force_head: bool = False
