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

import jax
from flax import nnx
import numpy as np

from dipm.utils.model_io import load_model, save_model


def test_model_can_be_saved_and_loaded_in_safetensors_format_correctly(
    setup_system_and_mace_model, tmp_path
):
    _, _, _, model_ff = setup_system_and_mace_model

    filepath = tmp_path / "model.safetensors"

    save_model(filepath, model_ff)
    loaded_model_ff = load_model(filepath)

    assert loaded_model_ff.config == model_ff.config

    assert jax.tree.map(
        np.shape, nnx.pure(nnx.state(loaded_model_ff, nnx.Param))
    ) == jax.tree.map(
        np.shape, nnx.pure(nnx.state(model_ff, nnx.Param))
    )

    original_params_flattened = nnx.to_flat_state(nnx.state(model_ff, nnx.Param))
    loaded_params_flattened = nnx.to_flat_state(nnx.state(loaded_model_ff, nnx.Param))
    for original_value, loaded_value in zip(
        original_params_flattened.leaves, loaded_params_flattened.leaves
    ):
        np.testing.assert_array_equal(original_value.value, loaded_value.value)
