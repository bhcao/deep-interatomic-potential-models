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

import json
import os
from typing import Type

import jax.numpy as jnp
from flax import nnx
from flax.nnx.traversals import unflatten_mapping
from safetensors.flax import save_file, safe_open

from dipm.data import DatasetInfo
from dipm.models import ForceFieldPredictor

PARAMETER_MODULE_DELIMITER = "."


def save_model(
    save_path: str | os.PathLike,
    model: ForceFieldPredictor,
) -> None:
    """Saves a force field model to a safetensors file.

    Args:
        save_path: The target path to the model file. Should have extension ".safetensors".
        model: The force field model to save.
               Must be passed as type :class:`~dipm.models.force_field.ForceField`.
    """
    hyperparams = {
        "dataset_info": model.dataset_info.model_dump_json(),
        "config": model.config.model_dump_json(),
        "predict_stress": 'true' if model.predict_stress else 'false', # for json to load
        "seed": str(model.seed)
    }

    _, state = nnx.split(model.force_model)

    params_flattened = {
        PARAMETER_MODULE_DELIMITER.join([str(i) for i in key_as_tuple]): array
        for key_as_tuple, array in nnx.to_flat_state(state)
    }

    save_file(params_flattened, save_path, metadata=hyperparams)


def _key2tuple(key: str) -> tuple:
    key_sep = key.split(PARAMETER_MODULE_DELIMITER)
    for i in range(len(key_sep)):
        if key_sep[i].isdigit():
            key_sep[i] = int(key_sep[i])
    return tuple(key_sep)


def load_model(
    model_type: Type[nnx.Module],
    load_path: str | os.PathLike,
) -> ForceFieldPredictor:
    """Loads a model from a safetensors file and returns it wrapped as a `ForceFieldPredictor`.

    Args:
        model_type: The model class that corresponds to the saved model.
        load_path: The path to the safetensors file to load.

    Returns:
        The loaded model wrapped
        as a :class:`~dipm.models.force_field.ForceFieldPredictor` object.
    """
    # load file
    params_raw = {}
    with safe_open(load_path, framework="flax") as f:
        for k in f.offset_keys():
            params_raw[_key2tuple(k)] = jnp.asarray(f.get_tensor(k))
        metadata = f.metadata()

    # load metadata
    hyperparams_raw = {}
    for k, v in metadata.items():
        hyperparams_raw[k] = json.loads(v)

    params = unflatten_mapping(params_raw)

    # Config(**hyperparams_raw["config"]) will be called by ForceModel.__init__
    model = model_type(
        config=hyperparams_raw["config"],
        dataset_info=DatasetInfo(**hyperparams_raw["dataset_info"]),
        rngs=nnx.Rngs(hyperparams_raw["seed"])
    )

    # load parameters
    graphdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, params)
    model = nnx.merge(graphdef, state)

    # `seed` is store only for next load
    return ForceFieldPredictor(model, hyperparams_raw["predict_stress"], seed=hyperparams_raw["seed"])
