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

import functools
import operator

import e3nn_jax as e3nn


def prod(xs):
    """From e3nn_jax/util/__init__.py."""
    return functools.reduce(operator.mul, xs, 1)


def tp_path_exists(arg_in1, arg_in2, arg_out):
    """Check if a tensor product path is viable.

    This helper function is similar to the one used in:
    https://github.com/e3nn/e3nn
    """
    arg_in1 = e3nn.Irreps(arg_in1).simplify()
    arg_in2 = e3nn.Irreps(arg_in2).simplify()
    arg_out = e3nn.Irrep(arg_out)

    for _multiplicity_1, irreps_1 in arg_in1:
        for _multiplicity_2, irreps_2 in arg_in2:
            if arg_out in irreps_1 * irreps_2:
                return True
    return False
