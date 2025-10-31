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

'''
In e3nn @ 0.4.0, the Wigner-D matrix is computed using Jd, while in e3nn @ 0.5.0,
it is computed using generators and matrix_exp causing a significant slowdown.
However, in e3nn_jax, `_wigner_D_from_angles` uses Jd for l <= 11 and matrix_exp
for l > 11, so it is well-optimized and there is no need for reimplement.
'''

from functools import lru_cache

from e3nn_jax._src.irreps import _wigner_D_from_angles
from e3nn_jax._src.s2grid import (
    _spherical_harmonics_s2grid, _normalization, _expand_matrix, _rollout_sh
)
from flax.typing import Dtype
from flax.struct import dataclass
import jax
import jax.numpy as jnp

from dipm.layers.escn.utils import order_mask, rescale_matrix


@lru_cache(maxsize=None)
def _get_s2grid_mat(
    lmax: int,
    res_beta: int,
    res_alpha: int,
    *,
    dtype: Dtype = jnp.float32,
    normalization: str = "integral",
) -> tuple[jax.Array, jax.Array]:
    r"""Modified `e3nn_jax._src.s2grid.to_s2grid` and `e3nn_jax._src.s2grid.from_s2grid` to act
    like `e3nn.o3.ToS2Grid` and `e3nn.o3.FromS2Grid`.

    Args:
        lmax (int): Maximum degree of the spherical harmonics
        res_beta (int): Number of points on the sphere in the :math:`\theta` direction
        res_alpha (int): Number of points on the sphere in the :math:`\phi` direction
        normalization ({'norm', 'component', 'integral'}): Normalization of the basis

    Returns:
        (to_grid_mat, from_grid_mat):
            Transform matrix from irreps to spherical grid and its inverse.
    """
    _, _, sh_y, sha, qw = _spherical_harmonics_s2grid(
        lmax, res_beta, res_alpha, quadrature="soft", dtype=dtype
    )
    # sh_y: (res_beta, l, |m|)
    sh_y = _rollout_sh(sh_y, lmax)

    m = jnp.asarray(_expand_matrix(range(lmax + 1)), dtype)  # [l, m, i]

    # construct to_grid_mat
    n_to = _normalization(lmax, normalization, dtype, "to_s2")
    sh_y_to = jnp.einsum("lmj,bj,lmi,l->mbi", m, sh_y, m, n_to)  # [m, b, i]
    to_grid_mat = jnp.einsum("mbi,am->bai", sh_y_to, sha)  # [beta, alpha, i]

    # construct from_grid_mat
    n_from = _normalization(lmax, normalization, dtype, "from_s2", lmax)
    sh_y_from = jnp.einsum("lmj,bj,lmi,l,b->mbi", m, sh_y, m, n_from, qw)  # [m, b, i]
    from_grid_mat = jnp.einsum("mbi,am->bai", sh_y_from, sha / res_alpha) # [beta, alpha, i]
    return to_grid_mat, from_grid_mat


@dataclass
class WignerMatrices:
    """Wigner-D matrix"""

    wigner: jax.Array
    wigner_inv: jax.Array

    def rotate(self, embedding):
        '''Rotate the embedding, l primary -> m primary.'''
        return jnp.matmul(self.wigner, embedding)

    def rotate_inv(self, embedding):
        '''Rotate the embedding by the inverse of rotation matrix, m primary -> to l primary.'''
        return jnp.matmul(self.wigner_inv, embedding)


class SO3Rotation:
    """
    Helper functions for Wigner-D rotations. Combined with `CoefficientMappingModule` to simplify.

    Args:
        lmax (int): Maximum degree of the spherical harmonics
    """

    def __init__(self, lmax: int, mmax: int, perm: jax.Array, *, dtype: Dtype = jnp.float32):
        self.lmax = lmax
        self.perm = perm

        self.mask = order_mask(lmax, mmax)

        # Compute the re-scaling for rotating back to original frame
        rotate_inv_rescale = rescale_matrix(lmax, mmax, dim=2, dtype=dtype)
        self.rotate_inv_rescale = rotate_inv_rescale[None, :, self.mask]

    def create_wigner_matrices(
        self,
        alpha: jax.Array,
        beta: jax.Array,
        gamma: jax.Array,
        scale: bool = True,
    ):
        '''Init the Wigner-D matrix for given euler angles.'''
        # Cache the Wigner-D matrices
        size = (self.lmax + 1) ** 2
        wigner = jnp.zeros([len(alpha), size, size], dtype=alpha.dtype)
        start = 0
        for lmax in range(self.lmax + 1):
            block = _wigner_D_from_angles(lmax, alpha, beta, gamma)
            end = start + block.shape[1]
            wigner = wigner.at[:, start:end, start:end].set(block)
            start = end

        # Mask the output to include only modes with m < mmax
        wigner = wigner[:, self.mask, :]
        wigner_inv = wigner.transpose((0, 2, 1))

        if scale:
            wigner_inv *= self.rotate_inv_rescale

        wigner = wigner[:, self.perm, :]
        wigner_inv = wigner_inv[:, :, self.perm]

        return WignerMatrices(wigner, wigner_inv)


class SO3Grid:
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        normalization: str = "component",
        resolution: int | None = None,
        *,
        dtype: Dtype = jnp.float32,
    ):
        mask = order_mask(lmax, mmax)

        lat_resolution = 2 * (lmax + 1)
        long_resolution = 2 * (mmax + 1 if lmax == mmax else mmax) + 1
        if resolution is not None:
            lat_resolution = resolution
            long_resolution = resolution

        # rescale last dimension based on mmax
        rescale_mat = rescale_matrix(lmax, mmax, dtype=jnp.float32)

        to_grid_mat, from_grid_mat = _get_s2grid_mat(
            lmax,
            lat_resolution,
            long_resolution,
            dtype=dtype,
            normalization=normalization,
        )
        self.to_grid_mat = (to_grid_mat * rescale_mat)[:, :, mask]
        self.from_grid_mat = (from_grid_mat * rescale_mat)[:, :, mask]

    def to_grid(self, embedding):
        '''Compute grid from irreps representation'''
        grid = jnp.einsum("bai, zic -> zbac", self.to_grid_mat, embedding)
        return grid

    def from_grid(self, grid):
        '''Compute irreps from grid representation'''
        embedding = jnp.einsum("bai, zbac -> zic", self.from_grid_mat, grid)
        return embedding

    def to_m_prime_format(self, perm: jax.Array) -> 'SO3Grid':
        """Operate on m primary mode so there is no need to permute the input/output."""
        new_self = self.__class__.__new__(self.__class__)
        new_self.to_grid_mat = self.to_grid_mat[:, :, perm]
        new_self.from_grid_mat = self.from_grid_mat[:, :, perm]
        return new_self
