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
        for i, entry in enumerate(data):
            group = f.create_group(str(i))

            group.create_dataset("positions", data=entry["positions"])
            group.create_dataset("elements", data=entry["elements"])

            if "energy" in entry: # only for test set
                group.attrs["energy"] = entry["energy"]
            if "forces" in entry: # only for test set
                group.create_dataset("forces", data=entry["forces"])
            if "stress" in entry:
                group.create_dataset("stress", data=entry["stress"])
            if "pbc" in entry:
                group.create_dataset("pbc", data=entry["pbc"])
            if "cell" in entry:
                group.create_dataset("cell", data=entry["cell"])
