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

import argparse
import logging
import json
from typing import TypeVar

import jax
import jax.numpy as jnp
import yaml
from flax import nnx
from safetensors.flax import safe_open

from dipm.data import GraphDatasetBuilder, DatasetInfo, ChemicalDatasetsConfig, create_datasets
from dipm.models import ForceFieldPredictor, KNOWN_MODELS
from dipm.loss import KNOWN_LOSSES
from dipm.training import (
    get_default_mlip_optimizer,
    OptimizerConfig,
    LineLogger,
    TensorBoardLogger,
    WandbLogger,
    TrainingLoop,
)
from dipm.training.training_io_handler import TrainingIOHandler
from dipm.utils.model_io import save_model, load_model
from dipm.typing import get_dtype

logger = logging.getLogger("dipm")

T = TypeVar('T')

def create_config(model: type[T], config: dict) -> T:
    '''Instantiate a pydantic model from a dictionary of values.'''
    config = {k: v for k, v in config.items() if k in model.model_fields}
    return model(**config)

def load_dataset_info(path: str) -> DatasetInfo:
    '''Load dataset info from pretrained model file.'''
    with safe_open(path, framework="flax") as f:
        metadata = f.metadata()
    dataset_info = json.loads(metadata['dataset_info'])
    return DatasetInfo(**dataset_info)

# pylint: disable=redefined-outer-name
def main(args):
    with open(args.config_file, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info("Loading dataset...")
    builder_config = create_config(GraphDatasetBuilder.Config, config["dataset"])
    if 'pretrained' not in config['model'] and builder_config.drop_unseen_elements:
        logger.warning(
            "`drop_unseen_elements` is set to True, but currently only works with pretrained "
            "models. The option will be forced to False."
        )
        builder_config.drop_unseen_elements = False

    # Load dataset info from file to skip computation
    dataset_info = None
    force_field = None

    # Load pretrained model before creating the dataset to overwrite the dataset_info.
    # That is essential for constructing correct atomic_species.
    if 'pretrained' in config['model']:
        dataset_info = load_dataset_info(config['model']['pretrained'])
    elif "info_file" in config["dataset"]:
        try:
            with open(config["dataset"]["info_file"], "r", encoding="utf-8") as f:
                dataset_info = create_config(DatasetInfo, json.load(f))
        except FileNotFoundError:
            logger.warning("Dataset info file %s not found.", config['dataset']['info_file'])

    reader_config = create_config(ChemicalDatasetsConfig, config["dataset"])
    if isinstance(reader_config.train_dataset_paths, dict):
        config["model"]["task_list"] = list(reader_config.train_dataset_paths.keys())
    builder = GraphDatasetBuilder(
        create_datasets(reader_config),
        builder_config,
        dataset_info,
    )
    # Compute all dataset information. If not called, you will get a warning.
    builder.prepare_datasets()
    logger.info("Dataset loaded.")

    should_parallelize = jax.device_count() > 1
    if "parallel" in config["train"]:
        should_parallelize = config["train"]["parallel"]
    elif should_parallelize:
        logger.info("Multiple devices detected, using parallel training. Set `parallel: false`"
                     " to disable.")

    if should_parallelize:
        train_set, validation_set, test_set = builder.get_splits(
            prefetch=should_parallelize,
            devices=jax.devices()
        )
    else:
        train_set, validation_set, test_set = builder.get_splits()

    # Create the model
    if 'pretrained' in config['model']:
        pretrained = config['model']['pretrained']
        origin_elements = set(dataset_info.atomic_energies_map.keys())
        target_elements = set(builder.dataset_info.atomic_energies_map.keys())
        elements_to_drop = None
        # This will happen if config['dataset']['drop_unseen_elements'] is True.
        if origin_elements != target_elements:
            elements_to_drop = origin_elements - target_elements
        force_field = load_model(
            pretrained,
            dtype=get_dtype(config["train"].get("dtype", None)),
            drop_force_head=config["train"].get("drop_force_head", False),
            elements_to_drop=elements_to_drop,
        )
        logger.info("Pretrained model loaded from %s.", pretrained)
    else:
        model_class = KNOWN_MODELS[config["model"]["target"].lower().replace('_', '')]
        force_model = model_class(
            create_config(model_class.Config, config["model"]),
            builder.dataset_info,
            dtype=get_dtype(config["train"].get("dtype", None)),
            rngs=nnx.Rngs(config["model"].get("seed", 42)),
        )
        force_field = ForceFieldPredictor(force_model)
        logger.info("Model created.")

    optimizer = get_default_mlip_optimizer(
        create_config(OptimizerConfig, config["train"])
    )

    loss_class = KNOWN_LOSSES[config["train"]["loss"].lower().replace('_', '').rstrip('loss')]
    if "stage_splits" in config["train"]:
        # Only support two stage training for now.
        num_epochs = config["train"]["num_epochs"]
        stage1 = int(config["train"]["stage_splits"][0] * num_epochs)

        energy_weights = config["train"]["energy_weights"]
        def energy_weight_schedule(epoch):
            return jnp.where(epoch <= stage1, energy_weights[0], energy_weights[1])
        forces_weights = config["train"]["forces_weights"]
        def forces_weight_schedule(epoch):
            return jnp.where(epoch <= stage1, forces_weights[0], forces_weights[1])

        loss = loss_class(
            energy_weight_schedule=energy_weight_schedule,
            forces_weight_schedule=forces_weight_schedule,
            extended_metrics=config["train"].get("extended_metrics", False),
        )
    else:
        loss = loss_class(
            extended_metrics=config["train"].get("extended_metrics", False),
        )

    io_handler = TrainingIOHandler(
        create_config(TrainingIOHandler.Config, config["train"]),
    )
    io_handler.attach_logger(LineLogger(logger))
    if "tensorboard" in config["train"]:
        io_handler.attach_logger(TensorBoardLogger(config["train"]["tensorboard"]))
    if "wandb" in config["train"]:
        io_handler.attach_logger(WandbLogger(config["train"]["wandb"]))

    training_loop = TrainingLoop(
        io_handler=io_handler,
        train_dataset=train_set,
        validation_dataset=validation_set,
        force_field=force_field,
        loss=loss,
        optimizer=optimizer,
        config=create_config(TrainingLoop.Config, config["train"]),
        should_parallelize=should_parallelize,
    )

    # Training loop
    training_loop.run()

    training_loop.test(test_set)

    optimized_force_field = training_loop.best_model
    save_model(config["train"]["save_path"], optimized_force_field)
    logger.info("Model saved to %s.", config["train"]["save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, force=True, format='%(levelname)s (%(name)s): %(message)s'
    )
    logger.setLevel(logging.INFO)

    main(args)
