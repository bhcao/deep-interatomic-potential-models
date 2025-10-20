# Copyright 2025 InstaDeep Ltd and Cao Bohan
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

from functools import lru_cache

from jax import Array
import jax.numpy as jnp
from flax.typing import Dtype


@lru_cache(maxsize=None)
def expand_index(lmax: int, mmax: int = None, vector_only: bool = False) -> Array:
    '''Expand coefficients from l values on different irreps to (l+1)**2 values on
    all elements.
    
    Args:
        lmax: maximum degree (l)
        mmax (optional): maximum order (m), defaults to `lmax`
        vector_only (optional): if True, only return coefficients on vector part,
            otherwise, return coefficients on both scalar and vector parts.
    '''
    if mmax is None:
        mmax = lmax
    lmin = 1 if vector_only else 0

    expand_index_list = []

    for lval in range(lmin, lmax + 1):
        length = min((2 * lval + 1), (2 * mmax + 1))
        expand_index_list.append(
            jnp.ones([length], dtype=jnp.int32) * (lval - 1)
        )

    return jnp.concat(expand_index_list)


@lru_cache(maxsize=None)
def order_mask(lmax: int, mmax: int, lmax_emb: int = None) -> Array:
    '''Compute the mask of orders less than or equal to `mmax` on IrrepsArray of
    `(1, ..., lmax)`.'''

    if lmax_emb is None:
        lmax_emb = lmax

    # Compute the degree (lval) and order (m) for each entry of the embedding
    m_harmonic_list = []
    l_harmonic_list = []
    for lval in range(lmax_emb + 1):
        m = jnp.arange(-lval, lval + 1)
        m_harmonic_list.append(jnp.abs(m))
        l_harmonic_list.append(jnp.ones_like(m) * lval)

    m_harmonic = jnp.concat(m_harmonic_list)
    l_harmonic = jnp.concat(l_harmonic_list)

    # Compute the indices of the entries to keep
    # We only use a subset of m components for SO(2) convolution
    return jnp.logical_and(l_harmonic <= lmax, m_harmonic <= mmax)


@lru_cache(maxsize=None)
def rescale_matrix(lmax: int, mmax: int, dim: int = 1, *, dtype: Dtype = jnp.float64) -> Array:
    '''Rescale matrix for masked entries based on `mmax`.'''

    size = (lmax + 1) ** 2
    matrix = jnp.ones([size] * dim, dtype=dtype)

    if lmax != mmax:
        for lval in range(lmax + 1):
            if lval <= mmax:
                continue
            start = lval ** 2
            length = 2 * lval + 1
            rescale_factor = jnp.sqrt(length / (2 * mmax + 1))
            slices = [slice(start, start + length)] * dim
            matrix = matrix.at[*slices].set(rescale_factor)

    return matrix
