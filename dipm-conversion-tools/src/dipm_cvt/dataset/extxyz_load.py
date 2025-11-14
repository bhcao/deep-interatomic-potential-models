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

from typing import TYPE_CHECKING

import numpy as np

from dipm_cvt._lazy_loader import LazyLoader

ase_io = LazyLoader(
    "ase_io", globals(), "ase.io",
    error_msg="`ase` is not installed but required for ExtXYZ dataset loading."
)
ase_data = LazyLoader(
    "ase_data", globals(), "ase.data",
    error_msg="`ase` is not installed but required for ExtXYZ dataset loading."
)
calculator = LazyLoader(
    "calculator", globals(), "ase.calculators.calculator",
    error_msg="`ase` is not installed but required for ExtXYZ dataset loading."
)

if TYPE_CHECKING:
    import ase.io as ase_io
    import ase.data as ase_data
    from ase.calculators import calculator


def load_extxyz_dataset(extxyz_path):
    """Read ExtXYZ dataset and return a list of Atoms objects."""

    atoms_list = ase_io.read(extxyz_path, index=":", format="extxyz")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    data = []
    for atoms in atoms_list:
        elements = np.array([ase_data.atomic_numbers[sym] for sym in atoms.symbols], dtype=np.int32)

        data_dict = {
            "positions": atoms.get_positions(),
            "elements": elements,
            "pbc": atoms.get_pbc(),
            "cell": np.array(atoms.get_cell()),
        }
        try: # only for test set
            data_dict["forces"] = atoms.get_forces() # eV / Ang
        except calculator.PropertyNotImplementedError:
            pass
        try: # only for test set
            data_dict["energy"] = atoms.get_potential_energy() # eV
        except calculator.PropertyNotImplementedError:
            pass
        try:
            data_dict["stress"] = atoms.get_stress(voigt=False) # eV / Ang^3, 3x3 matrix
        except calculator.PropertyNotImplementedError:
            pass

        data.append(data_dict)

    return data
