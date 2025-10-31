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

'''Borrowed from OpenQDC.'''

import logging
from pathlib import Path
import tempfile

import numpy as np

from dipm_cvt._lazy_loader import LazyLoader

openqdc_datasets = LazyLoader(
    "openqdc_datasets", globals(), "openqdc.datasets",
    error_msg="`openqdc` is not installed but required for OpenQDC download."
)

logger = logging.getLogger('dipm-cvt-cli')


def create_openqdc_dataset(
    url: str,
    download_dir: Path | None = None
) -> tuple[Path, list[dict[str, np.ndarray]]]:
    '''Create a dataset from OpenQDC.'''

    openqdc_dataset_map = {
        k.lower().replace("_", "").replace("-", ""): v
        for k, v in openqdc_datasets.AVAILABLE_DATASETS.items()
    }

    dataset_name = url.split("://")[1].lower().replace("_", "").replace("-", "")
    if dataset_name not in openqdc_dataset_map:
        logger.critical("Dataset %s is not available in OpenQDC.", dataset_name)

    dataset_class = openqdc_dataset_map[dataset_name]

    if download_dir is None:
        download_dir = Path(tempfile.mkdtemp())

    dataset = dataset_class(
        cache_dir=download_dir, energy_unit="ev", distance_unit="ang"
    )

    data: list[dict[str, np.ndarray]] = []
    for bunch in dataset:
        data_dict = {
            "positions": bunch.positions,
            "elements": bunch.atomic_numbers,
            "energy": bunch.energies.item(),
        }
        if bunch.get("forces") is not None:
            data_dict["forces"] = bunch.forces
        data.append(data_dict)

    return download_dir, data
