# Copyright 2025 InstaDeep Ltd
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

import matscipy.neighbours
import numpy as np

from dipm.utils.no_pbc_cell import get_no_pbc_cell


def _safe_matscipy_neighbour_list(**kwargs):
    """Forwards call to `matscipy.neighbours.neighbour_list`. Hence, the same keyword
    arguments as for that `matscipy` function are required. If call fails due
    to `np.linalg.LinAlgError` it is because the automatically computed cell inside
    matscipy has zeros on its diagonal. In that case (and only if PBC is false)
    we compute a proper cell and retry.
    """
    try:
        return matscipy.neighbours.neighbour_list(**kwargs)
    except np.linalg.LinAlgError:
        cell = kwargs.pop("cell")
        pbc = kwargs.pop("pbc")

        if cell is not None or any(pbc):
            raise ValueError(
                "Neighbour list creation with matscipy failed due to "
                "singular matrix inversion."
            ) from None

        positions = kwargs.pop("positions")
        cutoff = kwargs.pop("cutoff")
        cell, cell_origin = get_no_pbc_cell(positions, cutoff)
        return matscipy.neighbours.neighbour_list(
            **kwargs,
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            cell_origin=cell_origin,
        )


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    max_neighbors: int | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: np.ndarray | None = None,  # [3, 3]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the edge information for a given set of positions, including senders,
    receivers, and shift vectors.

    If ``pbc`` is ``None`` or ``(False, False, False)``, then the shifts will be
    returned as zero.
    This is the default behavior. The cell is None as default and as a result, matscipy
    will compute the minimal cell size needed to fit the whole system. See matscipy's
    documentation for more information.

    Args:
        positions: The position matrix.
        cutoff: The distance cutoff for the edges in Angstrom.
        max_neighbors: The maximum number of neighbors to consider for each atom.
        pbc: A tuple of bools representing if periodic boundary conditions exist in
             any of the spatial dimensions. Default is None, which means False in every
             direction.
        cell: The unit cell of the system given as a 3x3 matrix or as None (default),
              which means that matscipy will compute the minimal cell size needed to
              fit the whole system.

    Returns:
        A tuple of **senders** (starting indexes of atoms for each edge), **receivers**
        (ending indexes of atoms for each edge), and **shifts** (the shift vectors, see
        matscipy's documentation for more information. If PBCs are false,
        then we return shifts of zero).

    """
    if pbc is None:
        pbc = (False, False, False)

    if np.all(cell == 0.0):
        cell = None

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell is None or cell.shape == (3, 3)

    # See docstring of functions get_edge_relative_vectors() and
    # get_edge_vectors() on how senders and receivers are used
    neighbor_results = _safe_matscipy_neighbour_list(
        quantities="ijS" if max_neighbors is None else "ijSd",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )
    if max_neighbors is None:
        senders, receivers, senders_unit_shifts = neighbor_results
    else:
        senders, receivers, senders_unit_shifts, distances = neighbor_results
        # Filter out edges with more than max_neighbors
        nonmax_idx = []
        for i in range(len(positions)):
            idx_i = np.where(receivers == i)[0]
            # Sort neighbors by distance, remove edges larger than max_neighbors
            if len(idx_i) > max_neighbors:
                idx_sorted = np.argsort(distances[idx_i])[: max_neighbors]
                nonmax_idx.append(idx_i[idx_sorted])
            else:
                nonmax_idx.append(idx_i)
        nonmax_idx = np.concatenate(nonmax_idx)

        senders = senders[nonmax_idx]
        receivers = receivers[nonmax_idx]
        senders_unit_shifts = senders_unit_shifts[nonmax_idx]

    # If we are not having PBCs, then use shifts of zero
    shifts = senders_unit_shifts if any(pbc) else np.array([[0] * 3] * len(senders))

    # See docstring of functions get_edge_relative_vectors() and
    # get_edge_vectors() on how these return values are used
    return senders, receivers, shifts
