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

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class MappingCoefficients:
    '''Holds the mapping coefficients to reduce parameters.'''
    lmax: int
    mmax: int
    to_m: jnp.ndarray
    m_size: list[int]
    num_coefficients: int


def mapping_coefficients(lmax: int, mmax: int) -> MappingCoefficients:
    '''Return the mapping matrix from lval <--> m and size of each degree.'''

    # Compute the degree (lval) and order (m) for each entry of the embedding
    m_complex_list = []

    num_coefficients = 0
    for lval in range(lmax + 1):
        mmax_ = min(mmax, lval)
        m = jnp.arange(-mmax_, mmax_ + 1)
        m_complex_list.append(m)
        num_coefficients += len(m)

    m_complex = jnp.concat(m_complex_list, axis=0)

    # `self.to_m` moves m components from different L to contiguous index
    to_m = jnp.zeros([num_coefficients, num_coefficients])
    m_size = []

    # The following is implemented poorly - very slow. It only gets called
    # a few times so haven't optimized.
    offset = 0
    for m in range(mmax + 1):
        indices = jnp.arange(len(m_complex))
        # Real part
        idx_r = indices[m_complex == m]
        # Imaginary part
        idx_i = jnp.array([])
        if m != 0:
            idx_i = indices[m_complex == -m]

        # Add to the mapping matrix
        for idx_out, idx_in in enumerate(idx_r):
            to_m = to_m.at[idx_out + offset, idx_in].set(1.0)
        offset = offset + len(idx_r)

        m_size.append(len(idx_r))

        for idx_out, idx_in in enumerate(idx_i):
            to_m = to_m.at[idx_out + offset, idx_in].set(1.0)
        offset = offset + len(idx_i)

    return MappingCoefficients(lmax, mmax, jax.lax.stop_gradient(to_m), m_size, num_coefficients)


def pyg_softmax(src: jnp.ndarray, index: jnp.ndarray, num_segments: int) -> jnp.ndarray:
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

    src_max = jax.ops.segment_max(jax.lax.stop_gradient(src), index, num_segments)
    out = src - src_max[index]
    out = jnp.exp(out)
    out_sum = jax.ops.segment_sum(out, index, num_segments) + 1e-16
    out_sum = out_sum[index]

    return out / out_sum
