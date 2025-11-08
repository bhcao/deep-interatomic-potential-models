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

from dataclasses import replace
from typing import TypeAlias

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import nnx
import jraph
import numpy as np
from pydantic import BaseModel

from dipm.data import DatasetInfo
from dipm.data.helpers.edge_vectors import get_edge_relative_vectors
from dipm.models.force_model import ForceModel, PrecallInterface
from dipm.typing import Prediction

RelativeEdgeVectors: TypeAlias = np.ndarray
AtomicSpecies: TypeAlias = np.ndarray
Senders: TypeAlias = np.ndarray
Receivers: TypeAlias = np.ndarray
NodeEnergies: TypeAlias = np.ndarray


class ForceFieldPredictor(nnx.Module):
    """Flax module for a force field predictor.

    The apply function of this predictor returns the force field function used basically
    everywhere in the rest of the code base. This module is initialized from an
    already constructed MLIP model network module and a boolean whether to predict
    stress properties.

    Attributes:
        force_model: The MLIP network to use in this force field.
        predict_stress: Whether to predict stress properties. If false, only energies
                        and forces are computed.
        seed: The initialization seed for the parameters. Please keep same with rng.Rngs(seed).
              Default is 42.
    """

    def __init__(
        self,
        force_model: ForceModel,
        predict_stress: bool = False,
        seed: int = 42,
    ):
        """Only the `cutoff_distance` and `allowed_atomic_numbers` properties are subject
        to duck-typing in the simulation engine. Users are therefore free to provide
        any other force field callable that provides this simple interface.
        """
        self.force_model = force_model
        self.predict_stress = predict_stress
        self.seed = seed
        if isinstance(self.force_model, PrecallInterface):
            self.force_model.init_precall_key()

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        rngs: nnx.Rngs | None = None,
        ctx: dict | None = None,
    ) -> Prediction:
        """Returns a `Prediction` dataclass of properties based on an input graph.

        Note: The stress-related properties and "pressure" have not yet been thoroughly
        tested by us, so see them as an experimental feature for now.

        Args:
            graph: The input graph.
            rngs (optional): The random number generator for dropout. None for eval.
            ctx (optional): The context dictionary obtained from pre-calling the force model.

        Returns:
            The properties as a ``Prediction`` object that may include "energy",
            "forces", "stress", "stress_cell", "stress_forces", and "pressure".
            Only the first two exist if ``predict_stress=False`` is set for this module.
        """
        def cast_jraph(x: jax.Array):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(self.force_model.dtype)
            return x

        graph = jax.tree.map(cast_jraph, graph)

        if self.force_model.predict_forces:
            if self.predict_stress:
                pseudo_stress, (graph_energies, minus_forces) = nnx.grad(
                    self._compute_energy_and_forces, argnums=1, has_aux=True
                )(graph.nodes.positions, graph.globals.cell, graph, rngs, ctx)
            else:
                graph_energies, minus_forces = self._compute_energy_and_forces( # pylint: disable=W0632
                    graph.nodes.positions, graph.globals.cell, graph, rngs, ctx
                )
        else:
            if self.predict_stress:
                (minus_forces, pseudo_stress), graph_energies = nnx.grad(
                    self._compute_energy_and_forces, argnums=(0, 1), has_aux=True
                )(graph.nodes.positions, graph.globals.cell, graph, rngs, ctx)
            else:
                minus_forces, graph_energies = nnx.grad(
                    self._compute_energy_and_forces, argnums=0, has_aux=True
                )(graph.nodes.positions, graph.globals.cell, graph, rngs, ctx)

        result = Prediction(
            energy=graph_energies,  # [n_graphs,] energy per cell [eV]
            forces=-minus_forces,  # [n_nodes, 3] forces on each atom [eV / A]
        )

        if not self.predict_stress:
            return result

        # pylint: disable=used-before-assignment
        stress_results = self._compute_stress_results(
            graph, pseudo_stress, minus_forces
        )
        return replace(stress_results, energy=graph_energies, forces=-minus_forces)

    @staticmethod
    def _compute_stress_results(
        graph: jraph.GraphsTuple,
        pseudo_stress: np.ndarray,
        minus_forces: np.ndarray,
    ) -> Prediction:
        assert (
            graph.edges.shifts is not None
        ), "without shifts, the computed pseudo_stress is incorrect"

        det = jnp.linalg.det(graph.globals.cell)[:, None, None]  # [n_graphs, 1, 1]
        det = jnp.where(det > 0.0, det, 1.0)  # dummy graphs have det = 0

        # IMPORTANT NOTE:
        # These stress-related computations have not been thoroughly tested yet,
        # see them as an experimental feature for now.
        stress_cell = (
            jnp.transpose(pseudo_stress, (0, 2, 1)) @ graph.globals.cell
        )  # [n_graphs, 3, 3]
        stress_forces = e3nn.scatter_sum(
            jnp.einsum("iu,iv->iuv", minus_forces, graph.nodes.positions),
            nel=graph.n_node,
        )  # [n_graphs, 3, 3]
        viriel = stress_cell + stress_forces
        stress = -1.0 / det * viriel
        pressure = jnp.trace(stress, axis1=1, axis2=2)  # [n_graphs,]

        return Prediction(
            stress=stress,  # [n_graphs, 3, 3] stress tensor [eV / A^3]
            stress_cell=(
                -1.0 / det * stress_cell
            ),  # [n_graphs, 3, 3] stress tensor [eV / A^3]
            stress_forces=(
                -1.0 / det * stress_forces
            ),  # [n_graphs, 3, 3] stress tensor [eV / A^3]
            pressure=pressure,  # [n_graphs,] pressure [eV / A^3]
        )

    def _compute_energy_and_forces(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        graph: jraph.GraphsTuple,
        rngs: nnx.Rngs | None,
        ctx: dict | None,
    ):
        if graph.edges.shifts is None:
            assert graph.edges.displ_fun is not None
            vectors = graph.edges.displ_fun(
                positions[graph.receivers], positions[graph.senders]
            )
        else:
            vectors = get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=cell,
                n_edge=graph.n_edge,
            )

        if isinstance(self.force_model, PrecallInterface):
            if ctx is None:
                if hasattr(graph.globals, 'charge'):
                    charge = graph.globals.charge
                else:
                    charge = jnp.zeros_like(graph.n_node, dtype=jnp.int32)
                if hasattr(graph.globals,'spin'):
                    spin = graph.globals.spin
                else:
                    spin = jnp.zeros_like(graph.n_node, dtype=jnp.int32)
                if hasattr(graph.globals, 'dataset'):
                    dataset = graph.globals.dataset
                else:
                    dataset = jnp.zeros_like(graph.n_node, dtype=jnp.int32)

                ctx = self.force_model.precall(
                    node_species=graph.nodes.species,
                    charge=charge,
                    spin=spin,
                    n_node=graph.n_node,
                    dataset=dataset,
                    rngs=rngs,
                )
            node_energies = self.force_model(
                vectors, graph.nodes.species, graph.senders, graph.receivers, graph.n_node,
                rngs, ctx=ctx
            )
        else:
            node_energies = self.force_model(
                vectors, graph.nodes.species, graph.senders, graph.receivers, graph.n_node, rngs
            )  # [n_nodes, ]

        if self.force_model.predict_forces:
            node_energies, forces = node_energies

        node_padding_mask = jraph.get_node_padding_mask(graph)
        node_energies = node_energies * node_padding_mask
        assert node_energies.shape == (len(positions),), (
            f"model output needs to be an array of shape "
            f"(n_nodes, ) but got {node_energies.shape}"
        )

        graph_energies = e3nn.scatter_sum(node_energies, nel=graph.n_node)

        if self.force_model.predict_forces:
            forces = forces * node_padding_mask[:, None]
            assert forces.shape == (len(positions), 3), (
                f"model output needs to be an array of shape (n_nodes, 3) but got {forces.shape}"
            )

            if self.predict_stress:
                return jnp.sum(node_energies), graph_energies, -forces
            return graph_energies, -forces

        return jnp.sum(node_energies), graph_energies

    @property
    def cutoff_distance(self) -> float:
        """Cutoff distance in Angstrom the model was built for."""
        dataset_info = self.force_model.dataset_info
        return dataset_info.cutoff_distance_angstrom

    @property
    def allowed_atomic_numbers(self) -> set[int]:
        """Set of atomic numbers supported by the model."""
        dataset_info = self.force_model.dataset_info
        return set(dataset_info.atomic_energies_map.keys())

    @property
    def config(self) -> BaseModel:
        """Return configuration of the underlying MLIP model."""
        return self.force_model.config

    @property
    def dataset_info(self) -> DatasetInfo:
        """Return dataset info stored in the MLIP network."""
        return self.force_model.dataset_info

    def __hash__(self):
        """Simple hashing function to allow for jitting `self.__call__` directly."""
        return id(self)

    def __eq__(self, other):
        """Simple comparison based on IDs to allow for
        jitting `self.__call__` directly.
        """
        return id(other) == id(self)
