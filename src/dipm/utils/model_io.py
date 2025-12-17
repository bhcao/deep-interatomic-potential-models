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

import json
import os
import logging

import jax.numpy as jnp
from flax import nnx
from flax.nnx.traversals import unflatten_mapping
from flax.typing import Dtype
from safetensors.flax import save_file, safe_open

from dipm.data import DatasetInfo
from dipm.models import ForceFieldPredictor, KNOWN_MODELS
from dipm.models.force_model import ForceModel
from dipm.utils.params_filter import ParamsFilter

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


def _check_param_matches_and_drop_extra(
    params_raw: dict,
    model: ForceModel,
    strict: bool,
):
    state_dict = dict(nnx.to_flat_state(nnx.state(model, [nnx.Param, nnx.BatchStat])))
    if strict:
        for k, v in state_dict.items():
            if k not in params_raw:
                raise ValueError(f"Parameter {k} is missing from the saved file.")
            if v.shape != params_raw[k].shape:
                raise ValueError(f"Shape of parameter {k} is mismatched. Expected {v.shape}, "
                                 f"got {params_raw[k].shape}.")
        for k in params_raw:
            if k not in state_dict:
                raise ValueError(f"Extra parameter {k} is present in the saved file.")
    else:
        # remove extra or it will cause error when loading
        for k in list(params_raw.keys()):
            if k not in state_dict:
                del params_raw[k]


def load_model(
    load_path: str | os.PathLike,
    model_type: type[ForceModel] | None = None,
    dtype: Dtype | None = None,
    strict: bool = True,
    params_filters: ParamsFilter | list[ParamsFilter] | None = None,
) -> ForceFieldPredictor:
    """Loads a model from a safetensors file and returns it wrapped as a `ForceFieldPredictor`.

    Args:
        load_path: The path to the safetensors file to load.
        model_type (optional): The model class that corresponds to the saved model. If you are
            using a known model listed in `dipm.models.KNOWN_MODELS`, this argument can be optional
            and inferred from the saved metadata. Otherwise, it must be provided.
        dtype (optional): The dtype in computations. If not provided, it will be the same as the
            the parameters dtype.
        strict (optional): If True, raises an error if the parameters name and shape are mismatched.
        params_filters (optional): A function or a list of functions that takes the dataset_info,
            model_config, params_raw and model_type as input and returns the filtered dataset_info,
            model_config and params_raw. This can be used to modify the loaded parameters or
            metadata before creating the model.

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
    dataset_info = DatasetInfo(**hyperparams_raw["dataset_info"])

    if params_filters is not None:
        if not isinstance(params_filters, list):
            params_filters = [params_filters]
            for f in params_filters:
                dataset_info, model_config, params_raw = f(
                    dataset_info, model_config, params_raw, model_type
                )

    params_raw = {_key2tuple(k): v for k, v in params_raw.items()}
    params = unflatten_mapping(params_raw)

    model = model_type(config=model_config, dataset_info=dataset_info, dtype=dtype)

    _check_param_matches_and_drop_extra(params_raw, model, strict)

    # load parameters
    graphdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, params)
    model = nnx.merge(graphdef, state)

    # `seed` is store only for next load
    return ForceFieldPredictor(model, hyperparams_raw["predict_stress"])
