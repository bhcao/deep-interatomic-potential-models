# Copyright 2025 InstaDeep Ltd
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

from math import pi, sqrt
from collections.abc import Callable

import ase
import jax
import jax_md
import jraph
import numpy as np

from dipm.simulation.jax_md.jax_md_config import JaxMDSimulationConfig
from dipm.simulation.enums import SimulationType
from dipm.simulation.jax_md.states import SystemState

def create_units():
    """
    Physical constants and units derived from CODATA for converting to and from
    ase internal units. Copied from ase.units.
    """

    # CODATA 2014 taken from
    # http://arxiv.org/pdf/1507.07956.pdf
    u = {
        '_c': 299_792_458.,             # Exact
        '_mu0': 4.0e-7 * pi,            # Exact
        '_Grav': 6.674_08e-11,          # +/- 0.000_31e-11
        '_hplanck': 6.626_070_040e-34,  # +/- 0.000_000_081e-34
        '_e': 1.602_176_6208e-19,       # +/- 0.000_000_0098e-19
        '_me': 9.109_383_56e-31,        # +/- 0.000_000_11e-31
        '_mp': 1.672_621_898e-27,       # +/- 0.000_000_021e-27
        '_Nav': 6.022_140_857e23,       # +/- 0.000_000_074e23
        '_k': 1.380_648_52e-23,         # +/- 0.000_000_79e-23
        '_amu': 1.660_539_040e-27       # +/- 0.000_000_020e-27
    }

    # derived from the CODATA values
    u['_eps0'] = 1 / u['_mu0'] / u['_c']**2  # permittivity of vacuum
    u['_hbar'] = u['_hplanck'] / (2 * pi)  # Planck constant / 2pi, J s

    u['Ang'] = u['Angstrom'] = 1.0
    u['nm'] = 10.0
    u['Bohr'] = (4e10 * pi * u['_eps0'] * u['_hbar']**2 /
                 u['_me'] / u['_e']**2)  # Bohr radius

    u['eV'] = 1.0
    u['Hartree'] = (u['_me'] * u['_e']**3 / 16 / pi**2 /
                    u['_eps0']**2 / u['_hbar']**2)
    u['kJ'] = 1000.0 / u['_e']
    u['kcal'] = 4.184 * u['kJ']
    u['mol'] = u['_Nav']
    u['Rydberg'] = 0.5 * u['Hartree']
    u['Ry'] = u['Rydberg']
    u['Ha'] = u['Hartree']

    u['second'] = 1e10 * sqrt(u['_e'] / u['_amu'])
    u['fs'] = 1e-15 * u['second']

    u['kB'] = u['_k'] / u['_e']  # Boltzmann constant, eV/K

    u['Pascal'] = (1 / u['_e']) / 1e30  # J/m^3
    u['GPa'] = 1e9 * u['Pascal']
    u['bar'] = 1e5 * u['Pascal']

    u['Debye'] = 1.0 / 1e11 / u['_e'] / u['_c']
    u['alpha'] = (u['_e']**2 / (4 * pi * u['_eps0']) /
                  u['_hbar'] / u['_c'])  # fine structure constant
    u['invcm'] = (100 * u['_c'] * u['_hplanck'] /
                  u['_e'])  # cm^-1 energy unit

    # Derived atomic units that have no assigned name:
    # atomic unit of time, s:
    u['_aut'] = u['_hbar'] / (u['alpha']**2 * u['_me'] * u['_c']**2)
    # atomic unit of velocity, m/s:
    u['_auv'] = u['_e']**2 / u['_hbar'] / (4 * pi * u['_eps0'])
    # atomic unit of force, N:
    u['_auf'] = u['alpha']**3 * u['_me']**2 * u['_c']**3 / u['_hbar']
    # atomic unit of pressure, Pa:
    u['_aup'] = u['alpha']**5 * u['_me']**4 * u['_c']**5 / u['_hbar']**3

    u['AUT'] = u['second'] * u['_aut']

    # SI units
    u['m'] = 1e10 * u['Ang']  # metre
    u['kg'] = 1. / u['_amu']  # kilogram
    u['s'] = u['second']  # second
    u['A'] = 1.0 / u['_e'] / u['s']  # ampere
    # derived
    u['J'] = u['kJ'] / 1000  # Joule = kg * m**2 / s**2
    u['C'] = 1.0 / u['_e']  # Coulomb = A * s

    return {k: v for k, v in u.items() if not k.startswith('_')}


# Define all the expected symbols with dummy values so that introspection
# will know that they exist when the module is imported, even though their
# values are immediately overwritten.
# pylint: disable=invalid-name
(alpha, eV, fs, invcm, kB, kJ, kcal, kg, m, mol, nm, s, second, A, AUT, Ang, Angstrom,
 Bohr, C, Debye, GPa, Ha, Hartree, J, Pascal, bar, Ry, Rydberg) = [0.0] * 28

# Now update the module scope:
globals().update(create_units())


DUMMY_ARRAY = np.array([[0.0, 0.0, 0.0]])
DUMMY_CELL = np.array([[[0.0, 0.0, 0.0]] * 3])

TIMESTEP_CONVERSION_FACTOR = np.sqrt(kg * (kcal / mol) / J) * (m / Ang) * (fs / s)
TEMPERATURE_CONVERSION_FACTOR = kB / (kcal / mol)
KCAL_PER_MOL_PER_ELECTRON_VOLT = eV / (kcal / mol)
VELOCITY_CONVERSION_FACTOR = fs / TIMESTEP_CONVERSION_FACTOR

MINIMIZATION_PARAMETER_TIMESTEP_MAX_RATIO = 4
MINIMIZATION_PARAMETER_N_MIN = 5
MINIMIZATION_PARAMETER_F_INC = 1.1
MINIMIZATION_PARAMETER_F_DEC = 0.5
MINIMIZATION_PARAMETER_ALPHA_START = 0.1
MINIMIZATION_PARAMETER_F_ALPHA = 0.99


def batch_graph_with_one_dummy(
    system_state: SystemState,
    positions: np.ndarray,
    graph: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
    """Creates a batch of graphs out of a graph by adding one simple dummy graph.

    The dummy graph has just one node and one edge. Also, the positions of the input
    graph are updated with the given positions and the edges are also updated given
    the edges contained in the given system state.

    Args:
        system_state: The system state during the simulation.
        positions: The current positions of the system.
        graph: The graph of the system.

    Returns:
        The updated and batched graph.
    """
    neighbors = system_state.neighbors.update(positions)
    senders, receivers = neighbors.idx[1, :], neighbors.idx[0, :]

    new_positions = jax.lax.concatenate([positions, DUMMY_ARRAY], dimension=0)
    new_species = jax.lax.concatenate([graph.nodes.species, np.array([0])], dimension=0)

    num_nodes = int(graph.n_node[0])
    new_receivers = jax.lax.concatenate([receivers, np.array([num_nodes])], dimension=0)
    new_senders = jax.lax.concatenate([senders, np.array([num_nodes])], dimension=0)

    new_n_node = jax.lax.concatenate([graph.n_node, np.array([1])], dimension=0)
    new_n_edge = jax.lax.concatenate([graph.n_edge, np.array([1])], dimension=0)
    new_energy = jax.lax.concatenate(
        [graph.globals.energy, np.array([0.0])], dimension=0
    )
    new_weight = jax.lax.concatenate(
        [graph.globals.weight, np.array([0.0])], dimension=0
    )
    new_cell = jax.lax.concatenate([graph.globals.cell, DUMMY_CELL], dimension=0)

    return graph._replace(
        senders=new_senders,
        receivers=new_receivers,
        n_node=new_n_node,
        n_edge=new_n_edge,
        nodes=graph.nodes._replace(positions=new_positions, species=new_species),
        globals=graph.globals._replace(
            cell=new_cell,
            energy=new_energy,
            weight=new_weight,
        ),
    )


def init_simulation_algorithm(
    model_calculate_fun: Callable,
    shift_fun: Callable,
    sim_config: JaxMDSimulationConfig,
) -> tuple[Callable, Callable]:
    """Initializes the minimizer or MD integrator object of JAX-MD.

    Currently, for MD, the NVT-Langevin integrator is returned, and for energy
    minimization, the FIRE descent algorithm is used as the only options.

    Args:
        model_calculate_fun: The model calculate function outputting
                             either forces or energies.
        shift_fun: The shift function.
        sim_config: The pydantic config object for the JAX-MD simulation engine.

    Returns:
        A simulation init function and a simulation apply function used later to run
        the simulation.
    """
    if sim_config.simulation_type == SimulationType.MD:
        return jax_md.simulate.nvt_langevin(
            model_calculate_fun,
            shift_fun,
            kT=sim_config.temperature_kelvin * TEMPERATURE_CONVERSION_FACTOR,
            dt=sim_config.timestep_fs * TIMESTEP_CONVERSION_FACTOR,
        )

    start_timestep_fs = sim_config.timestep_fs * TIMESTEP_CONVERSION_FACTOR
    return jax_md.minimize.fire_descent(
        model_calculate_fun,
        shift_fun,
        dt_start=start_timestep_fs,
        dt_max=start_timestep_fs * MINIMIZATION_PARAMETER_TIMESTEP_MAX_RATIO,
        n_min=MINIMIZATION_PARAMETER_N_MIN,
        f_inc=MINIMIZATION_PARAMETER_F_INC,
        f_dec=MINIMIZATION_PARAMETER_F_DEC,
        alpha_start=MINIMIZATION_PARAMETER_ALPHA_START,
        f_alpha=MINIMIZATION_PARAMETER_F_ALPHA,
    )


def init_neighbor_lists(
    displacement_fun: Callable,
    positions: np.ndarray,
    cutoff_distance_angstrom: float,
    edge_capacity_multiplier: float,
) -> tuple[jax_md.partition.NeighborList, jax_md.partition.NeighborListFns]:
    """Initialize the neighbor lists objects for JAX-MD.

    Args:
        displacement_fun: The displacement function.
        positions: The positions of the system.
        cutoff_distance_angstrom: The graph cutoff distance in Angstrom.
        edge_capacity_multiplier: The edge capacity multiplier to decide how much
                                  padding is added to the neighbor lists.

    Returns:
        A tuple of the neighbor list object and the neighbor lists function object
        that JAX-MD needs for a simulation.
    """
    neighbor_fun = jax_md.partition.neighbor_list(
        displacement_fun,
        box=jax.numpy.nan,
        r_cutoff=cutoff_distance_angstrom,
        disable_cell_list=False,
        format=jax_md.partition.NeighborListFormat.Sparse,
        capacity_multiplier=edge_capacity_multiplier,
    )
    neighbors = neighbor_fun.allocate(positions)
    return neighbors, neighbor_fun


def get_masses(atoms: ase.Atoms) -> np.ndarray:
    """Returns the masses for a given set of atoms.

    Important note: this is currently just implemented as the ase.Atoms.get_masses()
    function which returns 1.008 for hydrogen instead of 1, etc. This may need to be
    adapted in the future, but for our H,C,N,O,S,P elements, the difference should be
    small.

    Args:
        atoms: An ase.Atoms object representing the molecule/system

    Returns:
        The atomic masses.
    """
    return atoms.get_masses()
