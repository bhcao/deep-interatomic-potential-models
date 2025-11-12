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

import argparse
import logging

from tqdm import tqdm

from dipm_cvt.cli.dataset_convert import (
    add_dataset_convert_args,
    run_dataset_convert,
)

DESCRIPTION = '''
Welcome to the dipm conversion tool! This tool can convert datasets in LMDB or
ExtXYZ format to HDF5 format and convert torch checkpoint of MACE to safetensors.
The tool supports parallel dataset conversion to speed up the conversion process.
'''


def get_argparser():
    '''Get the argument parser.'''

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "input_path",
        type=str,
        help="URL or path to the input datasets file / directory or model checkpoint",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output datasets file / directory or safetensors file",
    )
    main_group = parser.add_mutually_exclusive_group(required=True)
    main_group.add_argument(
        "--convert_dataset",
        "-d",
        action="store_true",
        help="Whether to convert datasets",
    )
    main_group.add_argument(
        "--convert_model",
        "-m",
        action="store_true",
        help="Whether to convert model checkpoint",
    )
    dataset_group = parser.add_argument_group('dataset conversion options')
    add_dataset_convert_args(dataset_group)
    return parser


class Handler(logging.StreamHandler):
    '''Handler for colorizing and tqdm compatible log messages.'''

    def __init__(self):
        super().__init__()
        self.fmt_dict = {
            logging.ERROR: "\033[35m%(log_msg)s\033[0m",
            logging.INFO: "%(log_msg)s",
            logging.DEBUG: "\033[2m%(log_msg)s\033[0m",
            logging.WARN: "\033[33m%(log_msg)s\033[0m",
            logging.WARNING: "\033[33m%(log_msg)s\033[0m",
            logging.CRITICAL: "\033[31m%(log_msg)s\033[0m",
        }

    def emit(self, record):
        msg = self.format(record)
        msg = self.fmt_dict.get(record.levelno, "%(log_msg)s") % {'log_msg': msg}
        tqdm.write(msg)
        if record.levelno == logging.CRITICAL:
            raise SystemExit(1)


def main():
    '''dipm_cvt_cli entry point.'''
    parser = get_argparser()
    args = parser.parse_args()

    # Set up logging
    handler = Handler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s (%(name)s): %(message)s"))

    logger = logging.getLogger('dipm-cvt-cli')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if args.convert_dataset:
        run_dataset_convert(args)

if __name__ == '__main__':
    main()
