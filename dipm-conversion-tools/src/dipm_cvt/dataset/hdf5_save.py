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

import h5py


def save_hdf5_dataset(data, hdf5_path):
    """Write a list of Atoms objects to an HDF5 dataset."""

    with h5py.File(hdf5_path, "w") as f:
        f.attrs["drop_last"] = len(data) % 2 == 1 # Why?
        main_grp = f.create_group("config_batch_0")
        for i, entry in enumerate(data):
            group = main_grp.create_group(f"config_{i}")

            group["atomic_numbers"] = entry["elements"]
            group["positions"] = entry["positions"]

            properties_subgrp = group.create_group("properties")
            if "energy" in entry: # only for test set
                properties_subgrp["energy"] = entry["energy"]
            if "forces" in entry: # only for test set
                properties_subgrp["forces"] = entry["forces"]
            if "stress" in entry:
                properties_subgrp["stress"] = entry["stress"]

            group["pbc"] = entry["pbc"] if "pbc" in entry else "None"
            group["cell"] = entry["cell"] if "cell" in entry else "None"
            group["weight"] = "None"

            group.create_group("property_weights")
            group["config_type"] = "None"
