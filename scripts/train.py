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

import argparse
import logging
import json

import yaml
import pydantic
from flax import nnx

from dipm.data import GraphDatasetBuilder, DatasetInfo
import dipm.models
from dipm.models import ForceFieldPredictor
import dipm.loss.loss
from dipm.training import get_default_mlip_optimizer, OptimizerConfig
from dipm.training.training_loggers import log_metrics_to_line
from dipm.training.training_io_handler import TrainingIOHandler
from dipm.training import TrainingLoop
from dipm.utils.model_io import save_model


def create_config(model: type[pydantic.BaseModel], config: dict):
    '''Instantiate a pydantic model from a dictionary of values.'''
    config = {k: v for k, v in config.items() if k in model.model_fields}
    return model(**config)


def main(args):
    with open(args.config_file, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logging.info("Loading dataset...")
    builder_config = create_config(GraphDatasetBuilder.Config, config["dataset"])

    # Load dataset info from file to skip computation
    if config["dataset"].get("info_file") is not None:
        try:
            with open(config["dataset"]["info_file"], "r", encoding="utf-8") as f:
                dataset_info = create_config(DatasetInfo, json.load(f))
            if builder_config.atomic_energies_map is None:
                builder_config.atomic_energies_map = dataset_info.atomic_energies_map
            if builder_config.avg_num_neighbors is None:
                builder_config.avg_num_neighbors = dataset_info.avg_num_neighbors
            if builder_config.avg_num_nodes is None:
                builder_config.avg_num_nodes = dataset_info.avg_num_nodes
            if builder_config.avg_r_min_angstrom is None:
                builder_config.avg_r_min_angstrom = dataset_info.avg_r_min_angstrom
        except FileNotFoundError:
            logging.warning("Dataset info file %s not found.", config['dataset']['info_file'])

    builder = GraphDatasetBuilder(
        create_config(GraphDatasetBuilder.ReaderConfig, config["dataset"]),
        builder_config
    )
    # Compute all dataset information. If not called, you will get a warning.
    builder.prepare_datasets()

    train_set, validation_set, test_set = builder.get_splits()
    logging.info("Dataset loaded.")

    # Create the model
    model_class = dipm.models.__dict__[config["model"]["target"]]
    force_model = model_class(
        create_config(model_class.Config, config["model"]),
        builder.dataset_info,
        rngs=nnx.Rngs(config["model"]["seed"]),
    )
    # This seed is for save_model()
    force_field = ForceFieldPredictor(force_model, seed=config["model"]["seed"])

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

        loss = dipm.loss.loss.__dict__[config["train"]["loss"]](
            energy_weight_schedule=energy_weight_schedule,
            forces_weight_schedule=forces_weight_schedule,
        )
    else:
        loss = dipm.loss.loss.__dict__[config["train"]["loss"]]()

    io_handler = TrainingIOHandler(
        create_config(TrainingIOHandler.Config, config["train"]),
    )
    io_handler.attach_logger(log_metrics_to_line)

    training_loop = TrainingLoop(
        io_handler=io_handler,
        train_dataset=train_set,
        validation_dataset=validation_set,
        force_field=force_field,
        loss=loss,
        optimizer=optimizer,
        config=create_config(TrainingLoop.Config, config["train"]),
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
    logging.getLogger("dipm").setLevel(logging.INFO)

    main(args)
