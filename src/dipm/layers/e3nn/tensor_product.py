# Copyright 2025 InstaDeep Ltd and Cao Bohan
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

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Dtype
import e3nn_jax as e3nn
from e3nn_jax import Irreps
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct

from dipm.layers.dtypes import promote_dtype


class FullyConnectedTensorProduct(nnx.Module):
    '''Flax module of FunctionalFullyConnectedTensorProduct'''

    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        irreps_out = Irreps(irreps_out).simplify()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        self.tensor_product = FunctionalFullyConnectedTensorProduct(
            irreps_in1, irreps_in2, irreps_out
        )
        key = rngs.params()
        self.weights = [
            nnx.Param(
                initializers.normal(stddev=ins.weight_std)(
                    key, ins.path_shape, param_dtype
                )
            )
            for ins in self.tensor_product.instructions
        ]
        self.dtype = dtype

    def __call__(
        self, x1: e3nn.IrrepsArray, x2: e3nn.IrrepsArray, **kwargs
    ) -> e3nn.IrrepsArray:
        weights = [w.value for w in self.weights]
        (x1, x2, weights) = promote_dtype((x1, x2, weights), dtype=self.dtype)

        leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        x1 = x1.broadcast_to(leading_shape + (-1,))
        x2 = x2.broadcast_to(leading_shape + (-1,))
        x1 = x1.rechunk(self.irreps_in1)
        x2 = x2.rechunk(self.irreps_in2)
        x1 = x1.remove_zero_chunks().simplify()
        x2 = x2.remove_zero_chunks().simplify()

        def helper(x1, x2):
            return self.tensor_product.left_right(weights, x1, x2, **kwargs)

        for _ in range(len(leading_shape)):
            helper_vmapped = e3nn.utils.vmap(helper)

        output = helper_vmapped(x1, x2)
        return output.rechunk(self.irreps_out)
