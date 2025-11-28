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

import e3nn_jax as e3nn
import flax
from flax import nnx
import jax.numpy as np

from dipm.models.mace.symmetric_contraction import SymmetricContraction


class TestSymmetricContraction:

    # test parameters
    rngs = nnx.Rngs(123)
    batch_size = 32
    irreps_in = "2x0e + 2x1o + 2x2e"
    # module arguments
    correlation = 3
    keep_irrep_out = "0e + 1o + 2e"
    num_species = 4

    def module_inputs(self) -> tuple[e3nn.IrrepsArray, np.ndarray]:
        """Prepare module inputs: (node_feats, species)."""
        nb = self.batch_size
        rep_in = e3nn.Irreps(self.irreps_in)
        node_feats = self.rngs.normal((nb, rep_in.dim))
        species = self.rngs.randint((nb,), 0, self.num_species)
        return (e3nn.IrrepsArray(rep_in, node_feats), species)

    def module(self) -> flax.linen.Module:
        """Prepare module."""
        return SymmetricContraction(
            self.batch_size,
            self.irreps_in,
            self.correlation,
            self.keep_irrep_out,
            self.num_species,
            rngs=self.rngs,
        )

    def test_symmetric_contraction(self):
        """Check that module runs without error."""
        module = self.module()
        inputs = self.module_inputs()
        out = module(*inputs)
        print(out.irreps)
        assert out.array.shape[0] == self.batch_size
