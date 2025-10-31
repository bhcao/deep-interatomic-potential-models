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

'''Borrowed from OpenQDC'''

import gzip
import logging
from pathlib import Path
import shutil
import tarfile
import zipfile

logger = logging.getLogger('dipm-cvt-cli')


def is_compressed_file(path: str | Path) -> bool:
    '''Check if a file or is compressed and whether the format is supported.'''

    if isinstance(path, Path):
        path = str(path)

    # Only .tar, .gz and .zip are supported though.
    suffixes = [".gz", ".zip", ".tar", ".tgz"]
    for suffix in suffixes:
        if path.endswith(suffix):
            return True

    suffixes = [".bz2", ".xz", ".rar", ".7z"]
    for suffix in suffixes:
        if path.endswith(suffix):
            logger.critical("Unsupported compressed file format: %s", suffix)
    return False


def _decompress_tar_gz(path_in: Path, path_out: Path):
    """
    Decompress a tar.gz file.
    
    Args:
        path_in (Path): Path to local file to decompress.
        path_out (Path): Path to decompressed directory.
    """
    with tarfile.open(path_in) as tar:
        all_names = tar.getnames()
        all_extracted = all((path_out / x).exists() for x in all_names)
        if not all_extracted:
            logger.info("Extracting archive %s to %s...", path_in, path_out)
            tar.extractall(path=path_out)
        else:
            logger.info("Archive %s already extracted.", path_in)


def _decompress_zip(path_in: Path, path_out: Path):
    """
    Decompress a zip file.

    Args:
        path_in (Path): Path to local file to decompress.
        path_out (Path): Path to decompressed directory.
    """
    with zipfile.ZipFile(path_in, "r") as zip_ref:
        all_names = zip_ref.namelist()
        all_extracted = all((path_out / x).exists() for x in all_names)
        if not all_extracted:
            logger.info("Extracting archive %s to %s...", path_in, path_out)
            zip_ref.extractall(path_out)
        else:
            logger.info("Archive %s already extracted.", path_in)


def _decompress_gz(path_in: Path, path_out: Path):
    """
    Decompress a gz file.

    Args:
        path_in (Path): Path to local file to decompress.
        path_out (Path): Path to decompressed directory.
    """
    file_path_out = path_out / path_in.name.replace(".gz", "")
    all_extracted = file_path_out.exists()
    if not all_extracted:
        logger.info("Extracting archive %s to %s...", path_in, file_path_out)
        with gzip.open(path_in, "rb") as f_in, open(file_path_out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        logger.info("Archive %s already extracted.", path_in)


def decompress(path_in: Path, path_out: Path):
    '''Decompress a file and return the path to the decompressed directory.
    
    Args:
        path_in (Path): Path to local file to decompress.
        path_out (Path): Path to decompressed directory.
    '''

    path_in_str = str(path_in)

    if path_in_str.endswith(".tar.gz") or path_in_str.endswith(".tgz"):
        _decompress_tar_gz(path_in, path_out)

    elif path_in_str.endswith(".zip"):
        _decompress_zip(path_in, path_out)

    elif path_in_str.endswith(".gz"):
        _decompress_gz(path_in, path_out)

    return path_out
