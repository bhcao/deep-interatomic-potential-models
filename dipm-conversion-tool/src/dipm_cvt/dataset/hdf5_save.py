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

import h5py


def save_hdf5_dataset(data, hdf5_path):
    """Write a list of Atoms objects to an HDF5 dataset."""

    with h5py.File(hdf5_path, "w") as f:
        f.attrs["drop_last"] = len(data) % 2 == 1 # Why?
        main_grp = f.create_group("config_batch_0")
        for i, entry in enumerate(data):
            group = main_grp.create_group(f"config_{i}")

            group["atomic_numbers"] = entry["atomic_numbers"]
            group["positions"] = entry["positions"]

            properties_subgrp = group.create_group("properties")
            if "energy" in entry: # only for test set
                properties_subgrp["energy"] = entry["energy"]
            if "forces" in entry: # only for test set
                properties_subgrp["forces"] = entry["forces"]
            if "stress" in entry:
                properties_subgrp["stress"] = entry["stress"]
            if "charges" in entry:
                properties_subgrp["charges"] = entry["charges"]
            if "total_charge" in entry:
                properties_subgrp["total_charge"] = entry["total_charge"]
            if "total_spin" in entry:
                properties_subgrp["total_spin"] = entry["total_spin"]
            if "dipole" in entry:
                properties_subgrp["dipole"] = entry["dipole"]

            group["pbc"] = entry["pbc"] if "pbc" in entry else "None"
            group["cell"] = entry["cell"] if "cell" in entry else "None"
            group["weight"] = "None"

            group.create_group("property_weights")
            group["config_type"] = "None"
