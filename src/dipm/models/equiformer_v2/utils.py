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


class AttntionActivationType(Enum):
    '''
    The options are as follows. Parameters not mentioned are False.

    Attributes:
        GATE: use_gate_act=True
        S2_SEP: use_sep_s2_act=True
        S2: else
    '''
    GATE = 'gate'
    S2_SEP ='s2_sep'
    S2 ='s2'


class FeedForwardType(Enum):
    '''
    The options are as follows. Parameters not mentioned are False.

    Attributes:
        GATE: Spectral atomwise, use_gate_act=True
        GRID: Grid atomwise, use_grid_mlp=True
        GRID_SEP: Grid atomwise, use_grid_mlp=True, use_sep_s2_act=True
        S2: S2 activation
        S2_SEP: S2 activation, use_sep_s2_act=True
    '''
    GATE = 'gate'
    GRID = 'grid'
    GRID_SEP = 'grid_sep'
    S2 = 's2'
    S2_SEP = 's2_sep'


def pyg_softmax(src: jax.Array, index: jax.Array, num_segments: int) -> jax.Array:
    r"""Computes a sparsely evaluated softmax referenced from torch_geometric.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.

    Returns:
        The softmax-ed tensor.
    """

    src_max = jax.ops.segment_max(src, index, num_segments)
    out = src - src_max[index]
    out = jnp.exp(out)
    out_sum = jax.ops.segment_sum(out, index, num_segments) + 1e-16
    out_sum = out_sum[index]

    return out / out_sum
