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

'''Borrowed from OpenQDC.'''

import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from dipm_cvt._lazy_loader import LazyLoader

openqdc_datasets = LazyLoader(
    "openqdc_datasets", globals(), "openqdc.datasets",
    error_msg="`openqdc` is not installed but required for OpenQDC download."
)

if TYPE_CHECKING:
    import openqdc.datasets as openqdc_datasets

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
