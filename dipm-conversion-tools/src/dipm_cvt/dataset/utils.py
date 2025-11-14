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
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger("dipm-cvt-cli")


def create_units(args):
    """Copied from ase.units.create_units() using the CODATA 2014 version."""

    energy_unit = args.energy_unit
    distance_unit = args.distance_unit

    u = {
        '_c': 299_792_458.,             # Exact
        '_mu0': 4.0e-7 * math.pi,       # Exact
        '_hplanck': 6.626_070_040e-34,  # +/- 0.000_000_081e-34
        '_e': 1.602_176_6208e-19,       # +/- 0.000_000_0098e-19
        '_me': 9.109_383_56e-31,        # +/- 0.000_000_11e-31
        '_Nav': 6.022_140_857e23,       # +/- 0.000_000_074e23
    }

    # derived from the CODATA values
    u['_eps0'] = 1 / u['_mu0'] / u['_c']**2  # permittivity of vacuum
    u['_hbar'] = u['_hplanck'] / (2 * math.pi)  # Planck constant / 2pi, J s

    bohr = (4e10 * math.pi * u['_eps0'] * u['_hbar']**2 /
                 u['_me'] / u['_e']**2)  # Bohr radius
    hartree = (u['_me'] * u['_e']**3 / 16 / math.pi**2 /
                    u['_eps0']**2 / u['_hbar']**2)
    kj_mol = 1000.0 / u['_e'] / u['_Nav']
    kcal_mol = 4.184 * kj_mol

    energy_map = {
        'eV': 1.0,
        'kcal/mol': kcal_mol,
        'Hartree': hartree,
        'kJ/mol': kj_mol,
    }
    length_map = {
        'Angstrom': 1.0,
        'Bohr': bohr,
    }

    # mapped units
    args.energy_unit = energy_map[energy_unit]
    args.distance_unit = length_map[distance_unit]
    args.force_unit = args.energy_unit / args.distance_unit
    args.stress_unit = args.energy_unit / args.distance_unit**3


def _ignore_new_path(new_path, args):
    '''Check whether to ignore the new path.'''
    if new_path.exists():
        if new_path.is_file():
            if args.overwrite_existing:
                new_path.unlink()
                return False
            if args.ignore_existing:
                return True
        logger.critical("Output path %s already exists.", new_path)
    return False


def get_single_map(input_path, output_path, args, add_suffix=False):
    """Create a mapping of a single input file or OpenQDC class to outputs.
    
    Args:
        add_suffix: Use parent directory name and add suffix.
    """

    if args.split:
        if args.size is None:
            args.size = 512 # default to 512 MB
        split_size = args.size * 1024 * 1024 # convert to bytes

        if output_path.is_file():
            logger.critical("Output path %s already exists.", output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        if isinstance(input_path, Path):
            input_size = input_path.stat().st_size
        elif isinstance(input_path, list):
            input_size = sum(len(d['elements']) for d in input_path) * 68 # approximate
        else: # This will never happen
            raise ValueError(f"Unsupported input type: {type(input_path)}")
        split = input_size // split_size
        if split == 0:
            split = 1 # if file is smaller than split_size
        new_paths_ = [output_path / f"data-{i:04d}.h5" for i in range(split)]
    else:
        new_paths_ = [output_path.with_suffix(".h5") if add_suffix else output_path]

    new_paths = []
    for new_path in new_paths_:
        if _ignore_new_path(new_path, args):
            new_paths.append(None)
            continue
        new_paths.append(new_path)

    # all files have been generated
    if all(path is None for path in new_paths):
        return None

    return ([input_path], new_paths)


def get_file_map(root, new_root, files, args):
    """Create a mapping of input files to output files."""

    # Filter files by suffix
    orig_files: list[Path] = []
    for file in files:
        if Path(file).suffix in set(['.extxyz', '.xyz', '.aselmdb', '.lmdb']):
            orig_path = Path(root) / file
            orig_files.append(orig_path.resolve())
    orig_files.sort()

    # Not merge files
    if not args.merge or len(orig_files) <= 1:
        for file in orig_files:
            out_map = get_single_map(file, new_root / file.stem, args, add_suffix=True)
            if out_map is not None:
                yield out_map
        return

    # Unlimited merge size
    if args.size is None:
        new_path = new_root / "data-0000.h5"
        if _ignore_new_path(new_path, args):
            return
        yield (orig_files, [new_path])
        return

    merge_size = args.size * 1024 * 1024 # convert to bytes
    file_sizes = [file.stat().st_size for file in orig_files]
    if sum(file_sizes) < merge_size:
        new_path = new_root / "data-0000.h5"
        if _ignore_new_path(new_path, args):
            return
        yield (orig_files, [new_path])
        return

    res_sizes = sum(file_sizes) - np.cumsum(file_sizes)

    # Divide files into groups of size <= merge_size
    group_count = 0
    group_size = 0
    group = []
    for file, size, res in zip(orig_files, file_sizes, res_sizes):
        group_size += size
        group.append(file)
        if group_size >= merge_size:
            # Add small residual file to the previous group
            if res < merge_size / 2 and res != 0:
                continue
            new_path = new_root / f"data-{group_count:04d}.h5"
            if _ignore_new_path(new_path, args):
                continue
            yield (group, [new_path])
            group_count += 1
            group_size = 0
            group = []

    if group:
        new_path = new_root / f"data-{group_count:04d}.h5"
        if _ignore_new_path(new_path, args):
            return
        yield (group, [new_path])
