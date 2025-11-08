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
import logging
from typing import Type

import jax.numpy as jnp
from flax import nnx
from flax.nnx.traversals import unflatten_mapping
from flax.typing import Dtype
from safetensors.flax import save_file, safe_open

from dipm.data import DatasetInfo
from dipm.models import ForceFieldPredictor, KNOWN_MODELS

PARAMETER_MODULE_DELIMITER = "."

logger = logging.getLogger('dipm')


def save_model(
    save_path: str | os.PathLike,
    model: ForceFieldPredictor,
) -> None:
    """Saves a force field model to a safetensors file.

    Args:
        save_path: The target path to the model file. Should have extension ".safetensors".
        model: The force field model to save.
               Must be passed as type :class:`~dipm.models.force_field.ForceFieldPredictor`.
    """
    hyperparams = {
        "dataset_info": model.dataset_info.model_dump_json(),
        "config": model.config.model_dump_json(),
        "predict_stress": 'true' if model.predict_stress else 'false', # for json to load
        "seed": str(model.seed)
    }

    # Add model type to hyperparams if model is known
    for k, v in KNOWN_MODELS.items():
        if isinstance(model.force_model, v):
            hyperparams["target"] = k
            break

    state = nnx.state(model.force_model, [nnx.Param, nnx.BatchStat])

    params_flattened = {
        PARAMETER_MODULE_DELIMITER.join([str(i) for i in key_as_tuple]): array
        for key_as_tuple, array in nnx.to_flat_state(state)
    }

    save_file(params_flattened, save_path, metadata=hyperparams)


def _key2tuple(key: str) -> tuple:
    key_sep = key.split(PARAMETER_MODULE_DELIMITER)
    for i, part in enumerate(key_sep):
        if part.isdigit():
            key_sep[i] = int(part)
    return tuple(key_sep)


def load_model(
    load_path: str | os.PathLike,
    model_type: Type[nnx.Module] | None = None,
    dtype: Dtype | None = None,
    drop_force_head: bool = True,
) -> ForceFieldPredictor:
    """Loads a model from a safetensors file and returns it wrapped as a `ForceFieldPredictor`.

    Args:
        load_path: The path to the safetensors file to load.
        model_type (optional): The model class that corresponds to the saved model. If you are
            using a known model listed in `dipm.models.KNOWN_MODELS`, this argument can be optional
            and inferred from the saved metadata. Otherwise, it must be provided.
        dtype (optional): The dtype in computations. If not provided, it will be the same as the
            the parameters dtype.
        drop_force_head (optional): If Ture, the head of the forces will be dropped if it exists
            in the saved model. Default is ``True`` because it is not recommended to use the head
            of the forces in inferece.

    Returns:
        The loaded model wrapped
        as a :class:`~dipm.models.force_field.ForceFieldPredictor` object.
    """
    # load file
    params_raw = {}
    with safe_open(load_path, framework="flax") as f:
        for k in f.offset_keys():
            params_raw[k] = jnp.asarray(f.get_tensor(k))
        metadata = f.metadata()

    # load metadata
    hyperparams_raw = {}
    for k, v in metadata.items():
        if k == "target":
            if v not in KNOWN_MODELS:
                raise ValueError(f"Model type {v} is unknown.")
            hyperparams_raw[k] = KNOWN_MODELS[v]
        else:
            hyperparams_raw[k] = json.loads(v)

    if model_type is None:
        if "target" not in hyperparams_raw:
            raise ValueError(
                f"Model type is neither present in the metadata of {load_path} nor provided."
            )
        model_type = hyperparams_raw["target"]

    model_config = model_type.Config(**hyperparams_raw["config"])

    # drop forces head if requested
    if drop_force_head and hasattr(model_config, "force_head") and model_config.force_head:
        prefix = model_type.force_head_prefix
        for k in list(params_raw.keys()):
            if k.startswith(prefix):
                del params_raw[k]
        model_config.force_head = False
        logger.info("Forces head has been dropped from the loaded model.")

    params = unflatten_mapping({_key2tuple(k): v for k, v in params_raw})

    # Config(**hyperparams_raw["config"]) will be called by ForceModel.__init__
    model = model_type(
        config=model_config,
        dataset_info=DatasetInfo(**hyperparams_raw["dataset_info"]),
        dtype=dtype,
        rngs=nnx.Rngs(hyperparams_raw["seed"])
    )

    # load parameters
    graphdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, params)
    model = nnx.merge(graphdef, state)

    # `seed` is store only for next load
    return ForceFieldPredictor(
        model, hyperparams_raw["predict_stress"], seed=hyperparams_raw["seed"]
    )
