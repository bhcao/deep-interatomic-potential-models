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

import logging
from pathlib import Path
import shutil
import socket
import tempfile
import urllib.error

from tqdm import tqdm

from dipm_cvt.dataset.decompress import (
    is_compressed_file,
    decompress,
)
from dipm_cvt.dataset.openqdc_ds import create_openqdc_dataset
from dipm_cvt._lazy_loader import LazyLoader

huggingface_hub = LazyLoader(
    "huggingface_hub", globals(), "huggingface_hub",
    error_msg="`huggingface_hub` is not installed but required for Hugging Face Hub download."
)
huggingface_hub_utils = LazyLoader(
    "huggingface_hub_utils", globals(), "huggingface_hub.utils",
    error_msg="`huggingface_hub` is not installed but required for Hugging Face Hub download."
)
requests = LazyLoader(
    "requests", globals(), "requests",
    error_msg="`requests` is not installed but required for web download."
)
gdown = LazyLoader(
    "gdown", globals(), "gdown",
    error_msg="`gdown` is not installed but required for Google Drive download."
)

logger = logging.getLogger('dipm-cvt-cli')


def _download(url: str, download_dir: Path | None = None) -> Path:
    '''Download a dataset from a URL.'''

    # Both check whether if URL is a compressed file and whether the format is supported.
    is_compress = is_compressed_file(url)

    if download_dir is not None and not is_compress:
        temp_dir = download_dir
    else:
        temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / Path(url).name

    try:
        logger.info("Downloading %s...", url)
        if "drive.google.com" in url:
            gdown.download(url, temp_file, quiet=False)
        else:
            r = requests.get(url, stream=True, timeout=300)
            file_size = None
            if 'content-length' in r.headers:
                file_size = int(r.headers['content-length'])
            if 'Content-Length' in r.headers:
                file_size = int(r.headers['Content-Length'])
            with open(temp_file, "wb") as f:
                pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=16384):
                    pbar.update(len(chunk))
                    if chunk:
                        f.write(chunk)
                pbar.close()

    except (socket.gaierror, urllib.error.URLError) as err:
        logger.critical("Could not download %s due to %s.", url, err)

    if is_compress:
        temp_compress = temp_dir
        if download_dir is not None:
            temp_dir = download_dir
        else:
            temp_dir = Path(tempfile.mkdtemp())
        decompress(temp_file, temp_dir)
        shutil.rmtree(temp_compress)

    return temp_dir


def _hf_download(url: str, download_dir: Path | None = None) -> Path:
    '''Download a dataset from Hugging Face Hub.'''

    is_compress = is_compressed_file(url)
    if download_dir is not None and not is_compress:
        temp_dir = download_dir
    else:
        temp_dir = Path(tempfile.mkdtemp())

    splits = url.split("://")[1].split("/", 2)
    if len(splits) < 2:
        logger.critical("Invalid Hugging Face Hub URL: %s", url)
    repo_id = f"{splits[0]}/{splits[1]}"

    try:
        if len(splits) == 2:
            huggingface_hub.snapshot_download(repo_id, repo_type="dataset", local_dir=temp_dir)
            return temp_dir

        filename = splits[2]
        huggingface_hub.hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=temp_dir)
    except huggingface_hub_utils.HfHubHTTPError as err:
        logger.critical("Could not download %s due to %s.", url, err)

    if is_compress:
        temp_compress = temp_dir
        if download_dir is not None:
            temp_dir = download_dir
        else:
            temp_dir = Path(tempfile.mkdtemp())
        decompress(temp_dir / filename, temp_dir)
        shutil.rmtree(temp_compress)

    return temp_dir


def _decompress(path: str, download_dir: Path | None = None) -> tuple[Path, bool]:
    '''Decompress only.
    
    Returns:
        The path to the decompressed directory and whether the input file was compressed.
    '''

    path = Path(path).resolve()

    if not is_compressed_file(path):
        return path, False

    if download_dir is not None:
        temp_dir = download_dir
    else:
        temp_dir = Path(tempfile.mkdtemp())
    decompress(path, temp_dir)
    return temp_dir, True


class DownloadManager:
    """A class to manage downloading datasets."""

    def __init__(self, url_or_path: str, download_dir: Path | None = None):
        if download_dir is not None:
            if not download_dir.is_dir():
                logger.critical("Download path %s is not a directory.", download_dir)
            download_dir.mkdir(exist_ok=True, parents=True)

        self._temp = download_dir is None
        self._openqdc_ds = None

        if url_or_path.startswith("https://") or url_or_path.startswith("http://"):
            self._path = _download(url_or_path, download_dir)
        elif url_or_path.startswith("hf://"):
            self._path = _hf_download(url_or_path, download_dir)
        elif url_or_path.startswith("openqdc://"):
            self._path, self._openqdc_ds = create_openqdc_dataset(url_or_path, download_dir)
        else:
            self._path, ok = _decompress(url_or_path, download_dir)
            if not ok: # self.path is input path
                self._temp = False

    def clear(self):
        '''Clear the downloaded dataset if it is temporary.'''
        if self._temp:
            shutil.rmtree(self._path)
            logger.info("Deleted temporary directory %s.", self._path)

    @property
    def dataset(self):
        '''Return the path to the downloaded dataset or OpenQDC dataset object.'''
        return self._path if self._openqdc_ds is None else self._openqdc_ds
