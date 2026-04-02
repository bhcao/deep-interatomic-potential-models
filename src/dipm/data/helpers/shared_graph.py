# Copyright 2025 Zhongguancun Academy
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

import numpy as np
from multiprocessing import shared_memory
from typing import Any
import jraph

from dipm.typing import GraphNodes, GraphEdges, GraphGlobals


class SharedGraphsTuple:
    """Shared memory wrapper for jraph.GraphsTuple."""

    def __init__(
        self,
        shm_name: str,
        metadata: dict[str, tuple[int, tuple[int, ...], str]],
        structure: dict[str, Any],
    ):
        self.shm_name = shm_name
        self.metadata = metadata
        self.structure = structure

    @classmethod
    def from_graph(cls, graph: jraph.GraphsTuple):
        """Create a SharedGraphsTuple from a jraph.GraphsTuple."""

        arrays: dict[str, np.ndarray] = {}

        def maybe_add(name, value):
            if isinstance(value, np.ndarray):
                arrays[name] = value

        maybe_add("senders", graph.senders)
        maybe_add("receivers", graph.receivers)
        maybe_add("n_node", graph.n_node)
        maybe_add("n_edge", graph.n_edge)

        if graph.nodes is not None:
            for field in GraphNodes._fields:
                v = getattr(graph.nodes, field)
                maybe_add(f"nodes.{field}", v)

        if graph.edges is not None:
            for field in GraphEdges._fields:
                v = getattr(graph.edges, field)
                maybe_add(f"edges.{field}", v)

        if graph.globals is not None:
            for field in GraphGlobals._fields:
                v = getattr(graph.globals, field)
                maybe_add(f"globals.{field}", v)

        metadata = {}
        offset = 0

        for name, arr in arrays.items():
            metadata[name] = (
                offset,
                arr.shape,
                str(arr.dtype),
            )

            offset += arr.nbytes

        shm = shared_memory.SharedMemory(create=True, size=offset)

        for name, arr in arrays.items():

            off, _, _ = metadata[name]

            size = arr.nbytes
            buf = shm.buf[off : off + size]

            dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=buf)
            dst[...] = arr

        # If not deleted, shm cannot be closed
        del dst
        del buf

        shm_name = shm.name
        shm.close()

        structure = dict(
            has_nodes=graph.nodes is not None,
            has_edges=graph.edges is not None,
            has_globals=graph.globals is not None,
        )

        if graph.edges is not None:
            structure["displ_fun"] = graph.edges.displ_fun

        return cls(shm_name, metadata, structure)

    def _attach(self):

        shm = shared_memory.SharedMemory(name=self.shm_name)

        arrays = {}

        for name, (offset, shape, dtype) in self.metadata.items():

            dtype = np.dtype(dtype)
            size = int(np.prod(shape)) * dtype.itemsize

            buf = shm.buf[offset : offset + size]

            arrays[name] = np.ndarray(shape, dtype=dtype, buffer=buf)

        return arrays, shm

    def to_graph(self) -> tuple[jraph.GraphsTuple, shared_memory.SharedMemory]:
        """Convert the SharedGraphsTuple to a jraph.GraphsTuple. Rememer to close the shared memory after use."""

        arrays, shm = self._attach()

        nodes = None
        if self.structure["has_nodes"]:

            values = []
            for field in GraphNodes._fields:
                key = f"nodes.{field}"
                values.append(arrays.get(key))

            nodes = GraphNodes(*values)

        edges = None
        if self.structure["has_edges"]:

            values = []
            for field in GraphEdges._fields:
                key = f"edges.{field}"

                if field == "displ_fun":
                    values.append(self.structure.get("displ_fun"))
                else:
                    values.append(arrays.get(key))

            edges = GraphEdges(*values)

        globals_ = None
        if self.structure["has_globals"]:

            values = []
            for field in GraphGlobals._fields:
                key = f"globals.{field}"
                values.append(arrays.get(key))

            globals_ = GraphGlobals(*values)

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globals_,
            senders=arrays["senders"],
            receivers=arrays["receivers"],
            n_node=arrays["n_node"],
            n_edge=arrays["n_edge"],
        )

        return graph, shm
