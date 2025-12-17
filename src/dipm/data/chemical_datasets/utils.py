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

import numpy as np

from dipm.data.chemical_system import ChemicalSystem
from dipm.data.helpers.neighborhood import get_neighborhood_stats


def filter_broken_system_and_get_stats(
    chemical_system: ChemicalSystem,
    cutoff: float,
    max_neighbours: int = None,
    max_set: set = None,
    max_num_nodes: int | None = None,
    max_total_edges: int | None = None,
    calc_stats: bool = False,
    calc_part: bool = False,
    calc_spices: bool = False,
) -> dict[str, np.ndarray | float | int] | None:
    """Filter out (return None) graph if it is empty, too large, or contain unseen elements,
    and return its statistics."""

    # 1. filter broken system

    species = set(chemical_system.atomic_numbers)
    if not (max_set is None or species.issubset(max_set)):
        return None

    num_nodes = len(chemical_system.positions)
    if num_nodes == 0 or (max_num_nodes is not None and num_nodes > max_num_nodes):
        return None

    # Atoms without edges are not included.
    num_neighbors, min_neighbor_distance = get_neighborhood_stats(
        chemical_system.positions,
        cutoff,
        max_neighbours,
        pbc=chemical_system.pbc,
        cell=chemical_system.cell,
    )
    if num_neighbors is None:
        return None

    total_edges = np.sum(num_neighbors)
    if total_edges == 0 or (max_total_edges is not None and total_edges > 2 * max_total_edges):
        return None

    # 2. get statistics

    output = {}
    if calc_part or calc_stats:
        output.update({
            'total_edges': total_edges,
            'num_nodes': num_nodes,
            'num_neighbors': num_neighbors,
        })

    if calc_spices or calc_stats:
        species = sorted(species)
        output['species'] = np.array(species)

    if calc_stats:
        species_count = np.array([
            np.count_nonzero(chemical_system.atomic_numbers == i)
            for i in species
        ], dtype=int)
        output.update({
            'min_neighbor_distance': min_neighbor_distance,
            'species_count': species_count,
            'energy': chemical_system.energy,
        })

    return output


# ase.chemical_symbols
CHEMICAL_SYMBOLS = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
