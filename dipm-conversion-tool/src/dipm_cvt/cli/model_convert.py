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

import logging
from pathlib import Path

from dipm_cvt.model.equiformer_v2 import convert_equiformer_v2

logger = logging.getLogger('dipm-cvt-cli')


def add_model_convert_args(parser):
    '''Add arguments to the parser.'''

    parser.add_argument(
        '--model_name',
        type=str,
        choices=['equiformer_v2'],
        default=None,
        help='The name of the model to convert.'
    )


def run_model_convert(args):
    '''Run model conversion with the given arguments.'''

    if args.model_name is None:
        logger.critical("Model name is not specified.")

    output_path = Path(args.output_path).absolute()
    if output_path.exists():
        if args.overwrite_existing:
            logger.warning("Existing file %s will be overwritten!", output_path)
            output_path.unlink()
        elif args.ignore_existing:
            logger.warning("Ignoring existing file %s.", output_path)
            return
        else:
            logger.critical("Output path %s already exists.", output_path)

    if args.model_name == 'equiformer_v2':
        convert_equiformer_v2(args.input_path, output_path)
        logger.info("Model conversion completed.")
