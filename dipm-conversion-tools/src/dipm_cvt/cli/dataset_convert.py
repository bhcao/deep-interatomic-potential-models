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

"""
Converts LMDB and ExtXYZ datasets to HDF5 datasets.
"""

import logging
import math
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
import os
from threading import Thread

from tqdm import tqdm

from dipm_cvt.dataset.download import DownloadManager
from dipm_cvt.dataset.utils import (
    create_units,
    get_single_map,
    get_file_map,
)
from dipm_cvt.dataset.lmdb_load import load_lmdb_dataset
from dipm_cvt.dataset.extxyz_load import load_extxyz_dataset
from dipm_cvt.dataset.hdf5_save import save_hdf5_dataset

logger = logging.getLogger('dipm-cvt-cli')


def add_dataset_convert_args(parser):
    '''Add arguments to the parser.'''

    parser.add_argument(
        "--ref_energy_path",
        type=str,
        default=None,
        help="Path to the reference energies file used in OC20 dataset",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to download dataset files, default to a temporary directory",
    )
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Whether to overwrite existing files",
    )
    existing_group.add_argument(
        "--ignore_existing",
        action="store_true",
        help="Whether to ignore existing files",
    )
    merge_split_group = parser.add_mutually_exclusive_group()
    merge_split_group.add_argument(
        "--merge_dir",
        action="store_true",
        help="Whether to merge subdirectory into a single HDF5 file",
    )
    merge_split_group.add_argument(
        "--split_file",
        action="store_true",
        help="Whether to split the input file into multiple HDF5 files",
    )
    parser.add_argument(
        "--merge_size",
        type=int,
        default=None,
        help="Total size of files to be merged into a HDF5 file (in MB), default to unlimited",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=512,
        help="Size of each split file (in MB), default to 512 MB",
    )
    parser.add_argument(
        "--energy_unit",
        type=str,
        choices=["eV", "kJ/mol", "kcal/mol", "Hartree", "Bohr"],
        default="eV",
        help="Energy unit used in the dataset, default to eV",
    )
    parser.add_argument(
        "--distance_unit",
        type=str,
        choices=["Angstrom", "Bohr"],
        default="Angstrom",
        help="Length unit used in the dataset, default to Angstrom",
    )


def convert_task(args, parallel=True):
    """Convert LMDB or ExtXYZ dataset to HDF5 dataset.
    
    Args:
        args: Tuple of (input_path, output_path, args, queue) if parallel
            else (input_path, output_path, args, pbar)
    """
    orig_path, new_path, args, queue = args

    if parallel:
        def log(msg, *args):
            queue.put(msg % args)
        update = queue.put
    else:
        log = logger.info
        update = queue.update # queue ia a pbar

    data = []
    for path in orig_path:
        if isinstance(path, list): # Converted dataset
            data.extend(path)
        elif path.suffix in [".lmdb", ".aselmdb"]:
            log("Loading LMDB dataset from %s...", path)
            if path.suffix == '.aselmdb':
                pyg_format = False
            else:
                pyg_format = True
            data.extend(load_lmdb_dataset(
                path, pyg_format, args.ref_energy_path, args.force_unit,
                args.energy_unit, args.stress_unit, args.distance_unit
            ))
        elif path.suffix in ['.extxyz', '.xyz']:
            log("Loading ExtXYZ dataset from %s...", path)
            data.extend(load_extxyz_dataset(path))
        else: # This will never happen
            raise NotImplementedError(f"File type of {path} is not supported.")
        update(1)

    split = math.ceil(len(data) / len(new_path))
    for i, path in enumerate(new_path):
        if path is None:
            continue
        start = i * split
        end = (i + 1) * split
        if i == len(new_path) - 1:
            end = len(data)
        data_part = data[start:end]
        log("Saving HDF5 dataset to %s...", path)
        save_hdf5_dataset(data_part, path)
        update(1)


def run_dataset_convert(args):
    '''Run dataset conversion with the given arguments.'''

    # Create default units from units name
    create_units(args)

    if args.overwrite_existing:
        logger.warning("Existing files will be overwritten!")

    # Create path mappings
    logger.info("Checking input and output paths...")

    # Download dataset if necessary
    downloader = DownloadManager(
        args.input_path,
        Path(args.download_dir).resolve() if args.download_dir is not None else None
    )
    input_path = downloader.dataset
    output_path = Path(args.output_path).absolute()

    if isinstance(input_path, Path) and not input_path.exists():
        logger.critical("Input path %s does not exist.", input_path)

    # list[tuple[list[Path | list[dict]], list[Path]]]
    file_map = []
    if isinstance(input_path, Path) and input_path.is_dir():
        if output_path.is_file():
            logger.critical(
                "Output path %s is a file, but input path is a directory.", output_path
            )
        output_path.mkdir(exist_ok=True, parents=True)
        for root, _, files in os.walk(input_path):
            new_root = output_path / Path(root).relative_to(input_path)
            new_root = new_root.resolve()
            new_root.mkdir(exist_ok=True, parents=True)

            for orig_path, new_path in get_file_map(root, new_root, files, args):
                file_map.append((orig_path, new_path))
    else: # input_path is a file or list of loaded data
        if not output_path.exists():
            output_path.parent.mkdir(exist_ok=True, parents=True)

        map_out = get_single_map(input_path, output_path, args)
        if map_out is not None:
            file_map.append(map_out)

    if len(file_map) == 0:
        logger.info("No files need to be converted.")
        return

    total_files_in = sum(len(files) for files, _ in file_map)
    total_files_out = sum(len([f for f in files if f is not None]) for _, files in file_map)
    logger.info(
        "Found %s dataset files, will convert to %s HDF5 files.",
        total_files_in, total_files_out
    )

    # Non-parallel conversion
    if len(file_map) == 1:
        pbar = tqdm(total=total_files_in + total_files_out, desc="Converting datasets")
        convert_task((*file_map[0], args, pbar), parallel=False)
        pbar.close()

        downloader.clear()
        logger.info("Dataset conversion complete.")
        return

    def progress_listener(queue, total):
        pbar = tqdm(total=total, desc="Converting datasets")
        count = 0
        while count < total:
            result = queue.get()
            if isinstance(result, str):
                logger.info(result)
            else:
                count += 1
                pbar.update(1)
        pbar.close()

    # Convert LMDB or ExtXYZ to HDF5
    with Manager() as manager:
        queue = manager.Queue()

        # For multiprocessing progress bar, both reading and writing files are counted as one task.
        listener = Thread(target=progress_listener, args=(queue, total_files_in + total_files_out))
        listener.start()

        # Dispatch tasks to worker processes
        args_list = [(inp, oup, args, queue) for inp, oup in file_map]

        with Pool(processes=min(cpu_count(), len(file_map))) as pool:
            pool.map(convert_task, args_list)

        listener.join()

    # Clear downloaded file if it is temporary
    downloader.clear()

    logger.info("Dataset conversion complete.")
