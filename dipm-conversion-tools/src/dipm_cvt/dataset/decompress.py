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

    if (path_in_str.endswith(".tar.gz") or path_in_str.endswith(".tgz")
        or path_in_str.endswith(".tar")):
        _decompress_tar_gz(path_in, path_out)

    elif path_in_str.endswith(".zip"):
        _decompress_zip(path_in, path_out)

    elif path_in_str.endswith(".gz"):
        _decompress_gz(path_in, path_out)

    return path_out
