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

import json
import logging
import os
import shutil
import time
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeAlias

import jax
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions

from dipm.data.dataset_info import DatasetInfo
from dipm.training.training_step import TrainingState
from dipm.utils.multihost import single_host_jax_and_orbax
from dipm.training.configs import TrainingIOHandlerConfig

PathLike: TypeAlias = str | os.PathLike
Source: TypeAlias = PathLike
Target: TypeAlias = PathLike
MODEL_SUBDIR_NAME = "model"
DATASET_INFO_FILENAME = "dataset_info.json"

_logger = logging.getLogger("dipm")


class CheckpointRestorationError(Exception):
    """Exception to be raised if issues occur during checkpoint restoration."""


class LogCategory(Enum):
    """Enum class for logging categories.

    These values provide a signal to a logging function what type of data is
    being logged.

    Attributes:
        BEST_MODEL: Information about the current best model is logged.
        TRAIN_METRICS: Metrics for the training set are logged.
        EVAL_METRICS: Metrics for the validation set are logged.
        TEST_METRICS: Metrics for the test set are logged.
        CLEANUP_AFTER_CKPT_RESTORATION: Allows the logger to clean itself up after a
                                        checkpoint has been restored.
        EPOCH_START: A separator to indicate the start of an epoch.
    """

    BEST_MODEL = 0
    TRAIN_METRICS = 1
    EVAL_METRICS = 2
    TEST_METRICS = 4
    CLEANUP_AFTER_CKPT_RESTORATION = 3
    EPOCH_START = 5


class TrainingIOHandler:
    """An IO handler class for the training loop.

    This handles checkpointing as well as specialized logging, e.g., to some external
    logger that a user can provide. If the config contains `None` for the local
    checkpointing directory, then this class will only do logging, but no
    checkpointing.
    """

    Config = TrainingIOHandlerConfig

    def __init__(
        self,
        config: TrainingIOHandlerConfig | None = None,
        data_upload_fun: Callable[[Source], Future | None] | None = None,
    ) -> None:
        """Constructor.

        Args:
            config: The training IO handler pydantic config. Can be `None` in which
                    case the default config will be used. Default is `None`.
            data_upload_fun: A data upload function to a remote storage.
                             This is optional, and set to None as default.
                             This function should just take in a source path, and then
                             the upload location can be user-defined within that
                             function. The function can be asynchronous in which case it
                             should return a Future.
        """
        self.config = config
        if self.config is None:
            self.config = TrainingIOHandlerConfig()

        self._local_model_output_dir = None
        if self.config.local_model_output_dir is not None:
            self._local_model_output_dir = Path(config.local_model_output_dir).resolve()

        self._data_upload_fun = data_upload_fun
        self.loggers = []
        self._future = None
        self.ckpt_manager = self._configure_checkpointing()

    def attach_logger(
        self, logger: Callable[[LogCategory, dict[str, Any], int], None]
    ) -> None:
        """Attaches one training loop logging function to the IO handler.

        The logging function must take in three parameter and should not return
        anything. The three parameters are a logging category which describes what
        type of data is logged (it is an enum), the data dictionary to log, and
        the current epoch number.

        Args:
            logger: The logging function to add.
        """
        self.loggers.append(logger)

    def log(
        self, category: LogCategory, to_log: dict[str, Any], epoch_number: int
    ) -> None:
        """Logs data via the logging functions stored in this class.

        Args:
            category: A logging category which describes what type of data is
                      logged (it is an enum)
            to_log: A data dictionary to log (typically, metrics).
            epoch_number: The current epoch number.
        """
        for logger in self.loggers:
            logger(category, to_log, epoch_number)

    def save_dataset_info(self, dataset_info: DatasetInfo) -> None:
        """Save the dataset information class to disk in JSON format.

        Will also upload with data upload function if it exists.

        Args:
            dataset_info: The dataset information class to save.
        """
        if self._local_model_output_dir is None:
            return

        _logger.debug("Saving/uploading dataset info...")

        start_time = time.perf_counter()
        local_json = self._local_model_output_dir / DATASET_INFO_FILENAME
        with local_json.open("w") as json_file:
            json.dump(json.loads(dataset_info.model_dump_json()), json_file, indent=4)
        if self._data_upload_fun is not None:
            self._data_upload_fun(local_json)

        _logger.debug(
            "Dataset info was saved and possibly uploaded in %.2f sec.",
            time.perf_counter() - start_time,
        )

    def save_checkpoint(self, training_state_dict: nnx.State, epoch_number: int) -> None:
        """Saves a model checkpoint to disk.

        Uses the data upload function as well if it exists.

        Args:
            training_state_dict: The nnx.State of the training state to save.
            epoch_number: The current epoch number.
        """
        if self._local_model_output_dir is None:
            return

        _logger.info("Saving checkpoint at epoch %s...", epoch_number)

        if self._future is not None:
            self._future.result()

        with single_host_jax_and_orbax():
            self.ckpt_manager.save(epoch_number, args=ocp.args.StandardSave(training_state_dict))

        if self._data_upload_fun is not None:
            self.ckpt_manager.wait_until_finished()
            _logger.info("Uploading checkpoint at epoch %s...", epoch_number)
            self._future = self._data_upload_fun(self.config.local_model_output_dir)

    def restore_training_state(self, training_state: TrainingState) -> TrainingState:
        """Restores a training state from disk locally.

        Note that if one wants to restore from a remote location, first download the
        state outside of this function.

        Args:
            training_state: An instance of training state, which will serve as a
                            template for the restoration.

        Returns:
            The restored training state.

        """
        if not self.config.restore_checkpoint_if_exists:
            return training_state

        if self._local_model_output_dir is None:
            raise CheckpointRestorationError(
                "Cannot restore training state as checkpointing is disabled."
            )

        # Skip when the directory is empty. It must be created by checkpointer.
        if not any((self._local_model_output_dir / 'model').iterdir()):
            return training_state

        start_time = time.perf_counter()
        epoch_to_restore = self.config.epoch_to_restore
        if epoch_to_restore is None:
            epoch_to_restore = self.ckpt_manager.latest_step()

        _logger.info("Restoring checkpoint from epoch %s.", epoch_to_restore)
        with single_host_jax_and_orbax():
            abstract_state = jax.tree_util.tree_map(
                ocp.utils.to_shape_dtype_struct,
                training_state.state_dict(ignore_cache=True),
            )
            ckpt = self.ckpt_manager.restore(
                epoch_to_restore,
                args=ocp.args.StandardRestore(abstract_state),
            )

        if self.config.restore_optimizer_state:
            _logger.debug("Restoring params and optimizer state.")
            nnx.update(training_state, ckpt)
        else:
            _logger.debug("Restoring params, resetting optimizer state.")
            ckpt = {"predictor": ckpt["predictor"]}
            nnx.update(training_state, ckpt)

        _logger.debug(
            "Checkpoint was restored in %.2f sec.", time.perf_counter() - start_time
        )

        return training_state

    def wait_until_finished(self) -> None:
        """Waits until the local checkpoint and `upload_fun` is finished due
        to their asynchronous nature. To be called at the end of a training run."""
        if self._local_model_output_dir is None:
            return

        self.ckpt_manager.wait_until_finished()
        if self._future is not None:
            self._future.result()

    def _configure_checkpointing(self) -> CheckpointManager | None:
        if self._local_model_output_dir is None:
            return None

        options = CheckpointManagerOptions(
            save_interval_steps=1,
            max_to_keep=self.config.max_checkpoints_to_keep,
            create=True,
            cleanup_tmp_directories=True,
        )

        self._handle_already_existing_checkpoint_dir()

        with single_host_jax_and_orbax():
            # Orbax presumes directory is shared and so only calls mkdir on process 0.
            return CheckpointManager(
                self._local_model_output_dir / MODEL_SUBDIR_NAME,
                options=options,
            )

    def _handle_already_existing_checkpoint_dir(self) -> None:
        if (
            self.config.restore_checkpoint_if_exists
            and self.config.clear_previous_checkpoints
        ):
            raise CheckpointRestorationError(
                "Cannot both restore and clear previous checkpoints."
            )

        # Skip when the directory does not exist.
        if not self._local_model_output_dir.exists():
            return

        if not self._local_model_output_dir.is_dir():
            raise CheckpointRestorationError(
                "Local checkpoint directory is not a directory."
            )

        if self.config.clear_previous_checkpoints:
            _logger.debug("Deleting local checkpointing directory...")
            shutil.rmtree(self._local_model_output_dir)
        elif self.config.restore_checkpoint_if_exists:
            _logger.debug(
                "Checkpointing directory exists locally and will be reused."
            )
