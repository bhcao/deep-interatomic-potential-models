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

import jax
import yaml
import pydantic
from flax import nnx

from dipm.data import GraphDatasetBuilder, DatasetInfo, ChemicalDatasetsConfig, create_datasets
import dipm.models
from dipm.models import ForceFieldPredictor
import dipm.loss
from dipm.training import (
    get_default_mlip_optimizer,
    OptimizerConfig,
    LineLogger,
    TensorBoardLogger,
    WandbLogger,
    TrainingLoop,
)
from dipm.training.training_io_handler import TrainingIOHandler
from dipm.utils.model_io import save_model
from dipm.typing import get_dtype

logger = logging.getLogger("dipm")


def create_config(model: type[pydantic.BaseModel], config: dict):
    '''Instantiate a pydantic model from a dictionary of values.'''
    config = {k: v for k, v in config.items() if k in model.model_fields}
    return model(**config)


def main(args):
    with open(args.config_file, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info("Loading dataset...")
    builder_config = create_config(GraphDatasetBuilder.Config, config["dataset"])

    # Load dataset info from file to skip computation
    dataset_info = None
    if config["dataset"].get("info_file") is not None:
        try:
            with open(config["dataset"]["info_file"], "r", encoding="utf-8") as f:
                dataset_info = create_config(DatasetInfo, json.load(f))
        except FileNotFoundError:
            logger.warning("Dataset info file %s not found.", config['dataset']['info_file'])

    reader_config = create_config(ChemicalDatasetsConfig, config["dataset"])
    builder = GraphDatasetBuilder(
        create_datasets(reader_config),
        builder_config,
        dataset_info,
    )
    # Compute all dataset information. If not called, you will get a warning.
    builder.prepare_datasets()
    logger.info("Dataset loaded.")

    should_parallelize = jax.device_count() > 1
    if config["train"].get("parallel", None) is not None:
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
    model_class = dipm.models.__dict__[config["model"]["target"]]
    force_model = model_class(
        create_config(model_class.Config, config["model"]),
        builder.dataset_info,
        dtype=get_dtype(config["train"].get("dtype", None)),
        rngs=nnx.Rngs(config["model"].get("seed", 42)),
    )
    # This seed is for save_model()
    force_field = ForceFieldPredictor(force_model)

    optimizer = get_default_mlip_optimizer(
        create_config(OptimizerConfig, config["train"])
    )

    if config["train"].get("stage_splits", None) is not None:
        # Only support two stage training for now.
        num_epochs = config["train"]["num_epochs"]
        stage1 = int(config["train"]["stage_splits"][0] * num_epochs)

        energy_weights = config["train"]["energy_weights"]
        def energy_weight_schedule(epoch):
            return energy_weights[0] if epoch <= stage1 else energy_weights[1]
        forces_weights = config["train"]["forces_weights"]
        def forces_weight_schedule(epoch):
            return forces_weights[0] if epoch <= stage1 else forces_weights[1]

        loss = dipm.loss.__dict__[config["train"]["loss"]](
            energy_weight_schedule=energy_weight_schedule,
            forces_weight_schedule=forces_weight_schedule,
        )
    else:
        loss = dipm.loss.__dict__[config["train"]["loss"]]()

    io_handler = TrainingIOHandler(
        create_config(TrainingIOHandler.Config, config["train"]),
    )
    io_handler.attach_logger(LineLogger(logger))
    if config["train"].get("tensorboard", None) is not None:
        io_handler.attach_logger(TensorBoardLogger(config["train"]["tensorboard"]))
    if config["train"].get("wandb", None) is not None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, force=True, format='%(levelname)s (%(name)s): %(message)s'
    )
    logger.setLevel(logging.INFO)

    main(args)
