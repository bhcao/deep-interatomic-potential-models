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

import functools
import io
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
import zlib

import numpy as np

from dipm_cvt._lazy_loader import LazyLoader

lmdb = LazyLoader(
    "lmdb", globals(), "lmdb",
    error_msg="`lmdb` is not installed but required for LMDB dataset."
)
torch_geometric = LazyLoader(
    "torch_geometric", globals(), "torch_geometric",
    error_msg="`torch_geometric` is not installed but required for OC20/OC22 LMDB dataset."
)
torch_geometric_data = LazyLoader(
    "torch_geometric_data", globals(), "torch_geometric.data",
    error_msg="`torch_geometric` is not installed but required for OC20/OC22 LMDB dataset."
)
json = LazyLoader(
    "json", globals(), "orjson",
    replace_name="json",
    warning_msg="`orjson` is not installed but recommended for faster JSON loading."
)

if TYPE_CHECKING:
    import lmdb
    import torch_geometric
    import torch_geometric.data as torch_geometric_data
    try:
        import orjson as json
    except ImportError:
        import json


def _pyg2_data_transform(data):
    """
    if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    we need to convert the data to the new format
    """
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return torch_geometric_data.Data(
            **{k: v for k, v in data.__dict__.items() if v is not None}
        )

    return data


class _LmdbPyGUnpickler(pickle.Unpickler):
    '''Safe unpickler for LMDB (PyG compatible) dataset.'''

    def __init__(self, file):
        super().__init__(file)
        self.allowed_packages = set(['collections.', 'torch.', 'torch_geometric.'])

    def find_class(self, module, name):
        module_with_dot = module + '.' # in case some malicious package has a same prefix
        for package in self.allowed_packages:
            if module_with_dot.startswith(package):
                return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unknown class tend to load: {module}.{name}.")


def _load_pyg_data(data_bytes, ref_energies=None):
    '''Load PyG Data object from LMDB bytes.'''
    with io.BytesIO(data_bytes) as f:
        data = _pyg2_data_transform(_LmdbPyGUnpickler(f).load())

    data_dict = {
        "positions": data.pos.numpy(),
        "elements": data.atomic_numbers.numpy().astype(np.int32),
    }
    if hasattr(data, "force"):
        data_dict["forces"] = data.force.numpy()
    if hasattr(data, "energy") or data.y is not None:
        data_dict["energy"] = float(data.energy if hasattr(data, "energy") else data.y)
    if hasattr(data, "stress"):
        data_dict["stress"] = data.stress.numpy()
    if hasattr(data, "cell"):
        data_dict["cell"] = data.cell.numpy()
    if hasattr(data, "pbc"):
        data_dict["pbc"] = data.pbc.numpy()

    if ref_energies is not None and data_dict.get("energy") is not None:
        data_dict["energy"] += ref_energies[f'random{data.sid}']

    return data_dict


def _load_json_data(data_bytes):
    '''Load JSON data from LMDB bytes.'''
    data_dict_old = json.loads(zlib.decompress(data_bytes))

    data_dict = {
        "positions": np.array(data_dict_old["positions"]),
        "elements": np.array(data_dict_old["numbers"], dtype=np.int32),
    }
    if data_dict_old.get("forces") is not None:
        data_dict["forces"] = np.array(data_dict_old["forces"])
    if data_dict_old.get("energy") is not None:
        data_dict["energy"] = data_dict_old["energy"]
    if data_dict_old.get("stress") is not None:
        data_dict["stress"] = np.array(data_dict_old["stress"])
    if data_dict_old.get("pbc") is not None:
        data_dict["pbc"] = np.array(data_dict_old["pbc"])
    if data_dict_old.get("cell") is not None:
        data_dict["cell"] = np.array(data_dict_old["cell"])

    return data_dict


def load_lmdb_dataset(
    lmdb_path: str | Path,
    pyg_format: bool = False,
    ref_energy_path: str | Path | None = None,
    force_unit: float = 1.0,
    energy_unit: float = 1.0,
    stress_unit: float = 1.0,
    distance_unit: float = 1.0,
) -> list[dict[str, np.ndarray]]:
    '''Load LMDB dataset and return a list of dictionaries.
    
    Args:
        lmdb_path (Path): Path to LMDB dataset.
        pyg_format (bool): Whether in PyG format (OC20/OC22) or json format (OMat24/OMol25).
        ref_energy_path (Path, optional): Path to reference energies file used in OC20 dataset.
        force_unit (float): Force unit conversion factor.
        energy_unit (float): Energy unit conversion factor.
        stress_unit (float): Stress unit conversion factor.
        distance_unit (float): Length unit conversion factor.
    '''

    # Not used, as we will iterate over all data
    ignored_keys = set([b'length', b'deleted_ids', b'nextid', b'metadata'])

    if ref_energy_path is not None:
        with open(ref_energy_path, "rb") as f:
            ref_energies = pickle.load(f)
    else:
        ref_energies = None

    if pyg_format:
        data_loader = functools.partial(_load_pyg_data, ref_energies=ref_energies)
    else:
        data_loader = _load_json_data

    def unit_convert(data_dict):
        if distance_unit != 1.0:
            data_dict["positions"] *= distance_unit
            if "cell" in data_dict:
                data_dict["cell"] *= distance_unit
        if energy_unit != 1.0:
            if "energy" in data_dict:
                data_dict["energy"] *= energy_unit
        if force_unit != 1.0: # When force_unit != 1.0, stress_unit must also != 1.0
            if "forces" in data_dict:
                data_dict["forces"] *= force_unit
            if "stress" in data_dict:
                data_dict["stress"] *= stress_unit

        return data_dict

    data: list = []
    with lmdb.open(str(lmdb_path), subdir=False) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                if key in ignored_keys:
                    continue
                data.append(unit_convert(data_loader(value)))

    return data
