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
