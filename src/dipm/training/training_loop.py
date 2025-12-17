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

import logging
import time
from functools import partial
from collections.abc import Callable
from typing import TypeAlias

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from tqdm import tqdm

from dipm.data import DataLoader
from dipm.models import ForceFieldPredictor
from dipm.loss import Loss
from dipm.training.optimizer import EMATracker
from dipm.training.evaluation import (
    make_evaluation_step, run_evaluation, convert_mse_to_rmse_in_logs
)
from dipm.training.training_io_handler import LogCategory, TrainingIOHandler
from dipm.training.loggers.command_line import LineLogger
from dipm.training.configs import TrainingLoopConfig
from dipm.training.training_step import TrainingState, TrainingStateVar, make_train_step

TrainingStepFun: TypeAlias = Callable[
    [TrainingState, jraph.GraphsTuple],
    tuple[TrainingState, dict],
]

logger = logging.getLogger("dipm")


def _count_parameters(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))


class _TrainingLog:
    def __init__(self, num_steps):
        self.last_log_step = num_steps
        self.epoch_last_log_step = num_steps
        self.metrics: list[dict] = []
        self.start_time = time.perf_counter()
        self.epoch_start_time = self.start_time

    def append(self, metrics: dict) -> None:
        '''Appends metrics of a training.'''
        self.metrics.append(metrics)

    def get(self, num_steps: int):
        '''Returns the metrics of the training.'''
        end_time = time.perf_counter()
        steps = num_steps - self.last_log_step
        time_per_step = (end_time - self.start_time) / steps
        metrics = {}
        for metric_name in self.metrics[0].keys():
            metrics[metric_name] = np.mean([m[metric_name] for m in self.metrics])
        metrics["seconds_per_step"] = time_per_step
        # Update
        self.last_log_step = num_steps
        self.start_time = end_time
        self.metrics = []
        return metrics

    def get_epoch(self, num_steps: int):
        '''Returns the time and step used of the current epoch.'''
        end_time = time.perf_counter()
        steps = num_steps - self.epoch_last_log_step
        total_time = end_time - self.epoch_start_time
        metrics = {}
        metrics["runtime_in_minutes"] = total_time / 60
        metrics["steps_per_epoch"] = steps
        # Update
        self.epoch_last_log_step = num_steps
        self.epoch_start_time = end_time
        return metrics


class TrainingLoop:
    """Training loop class.

    It implements only the loop based on its inputs but does not construct any
    auxiliary objects within it. For example, the model, dataset, and optimizer must
    be passed to this function from the outside.

    Attributes:
        training_state: The training state.
        best_model (ForceFieldPredictor): The current state of the force field model
            with the best parameters so far.
    """

    Config = TrainingLoopConfig

    def __init__(
        self,
        train_dataset: DataLoader,
        validation_dataset: DataLoader,
        force_field: ForceFieldPredictor,
        loss: Loss,
        optimizer: optax.GradientTransformation,
        config: TrainingLoopConfig,
        io_handler: TrainingIOHandler | None = None,
    ) -> None:
        """Constructor.

        Args:
            train_dataset: The training dataset as either a GraphDataset or
                           a PrefetchIterator.
            validation_dataset: The validation dataset as either a GraphDataset or
                                a PrefetchIterator.
            force_field: The force field model holding at least the initial parameters
                         and a dataset info object.
            loss: The loss, which it is derived from the `Loss` base class.
            optimizer: The optimizer (based on optax).
            config: The training loop pydantic config.
            io_handler: The IO handler which handles checkpointing
                        and (specialized) logging. This is an optional argument.
                        The default is `None`, which means that a default IO handler
                        will be set up which does not include checkpointing but some
                        very basic metrics logging.
        """
        self.should_parallelize = train_dataset.devices is not None

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.force_field = force_field
        self.config = config

        self.num_batches, self.total_num_graphs, self.total_num_nodes = (
            self._get_number_of_batches_graphs_and_nodes(self.train_dataset, "train")
        )

        self.extended_metrics = (
            True if not hasattr(loss, "extended_metrics") else loss.extended_metrics
        )
        self.io_handler = io_handler
        if self.io_handler is None:
            self.io_handler = TrainingIOHandler()
            self.io_handler.attach_logger(LineLogger(logger))

        self.io_handler.save_dataset_info(self.force_field.dataset_info)

        self._loss_train = partial(loss, eval_metrics=False)
        self._loss_eval = partial(loss, eval_metrics=True)

        self._prepare_training_state(optimizer)
        # Note: Because we shuffle the training data between epochs, the following
        # value may slightly fluctuate during training, however, we assume
        # it being fixed, which is a solid approximation for datasets of typical size.
        _avg_n_graphs_train = self.total_num_graphs / self.num_batches
        self.training_step = make_train_step(
            self._loss_train,
            _avg_n_graphs_train,
            config.num_gradient_accumulation_steps,
            self.should_parallelize,
        )
        self.metrics = None
        num_batches_val, num_graphs_val, _ = self._get_number_of_batches_graphs_and_nodes(
            self.validation_dataset, "validation"
        )
        _avg_n_graphs_validation = num_graphs_val / num_batches_val
        # Clone of force_field, will be updated every training round.
        self.best_model = nnx.clone(force_field)
        self.eval_step = make_evaluation_step(
            self._loss_eval,
            _avg_n_graphs_validation,
            self.should_parallelize,
        )

        self.best_evaluation_step = -1
        self.best_evaluation_loss = float("inf")
        self.best_evaluation_epoch = -1

        self.steps_per_epoch = self.num_batches
        if self.should_parallelize:
            self.steps_per_epoch = (
                self.num_batches // len(jax.devices())
            ) // config.num_gradient_accumulation_steps
        self.epoch_number = self._get_epoch_number_from_training_state()

        logger.debug(
            "Training loop: Number of batches has been set to: %s", self.num_batches
        )
        logger.debug(
            "Training loop: Steps per epoch has been set to: %s", self.steps_per_epoch
        )

    def run(self, rngs: nnx.Rngs | None = None) -> None:
        """Runs the training loop.

        The final training state can be accessed via its member variable.

        Args:
            rngs: The random number generators for training. Only used if the model contains
                  dropout or other stochastic layers. Default to `nnx.Rngs(42)`.
        """
        if rngs is None:
            rngs = nnx.Rngs(42)

        logger.info("Starting training loop...")

        # May not be zero if restored from checkpoint
        if self.epoch_number > 0:
            self.io_handler.log(
                LogCategory.CLEANUP_AFTER_CKPT_RESTORATION, {}, self.epoch_number
            )

        if self.epoch_number == 0 and self.config.run_eval_at_start:
            logger.debug("Running initial evaluation...")
            start_time = time.perf_counter()
            self._run_evaluation()
            logger.debug(
                "Initial evaluation done in %.2f sec.", time.perf_counter() - start_time
            )

        log = _TrainingLog(self._get_num_steps_from_training_state())
        while self.epoch_number < self.config.num_epochs:
            self.epoch_number += 1
            self.io_handler.log(LogCategory.EPOCH_START, {}, self.epoch_number)
            t_before_train = time.perf_counter()
            self._run_training_epoch(rngs, log)
            logger.debug(
                "Parameter updates of epoch %s done, running evaluation next.",
                self.epoch_number,
            )
            t_after_train = time.perf_counter()
            self._run_evaluation(log)
            t_after_eval = time.perf_counter()

            logger.debug(
                "Epoch %s done. Time for parameter updates: %.2f sec.",
                self.epoch_number,
                t_after_train - t_before_train,
            )
            logger.debug("Time for evaluation: %.2f sec.", t_after_eval - t_after_train)

        self.io_handler.wait_until_finished()

        logger.info("Training loop completed.")

    def _run_training_epoch(self, rngs: nnx.Rngs, log: _TrainingLog) -> None:
        self.force_field.train()

        log_interval = self.config.log_interval or self.steps_per_epoch
        for batch in self.train_dataset:
            _metrics = self.training_step(
                self.training_state, batch, rngs, self.epoch_number
            )
            log.append(jax.device_get(_metrics))

            num_steps = self._get_num_steps_from_training_state()
            if num_steps % log_interval == 0:
                metrics = log.get(num_steps)
                self._log_after_training_steps(metrics, self.epoch_number, num_steps)

    def _run_evaluation(self, log: _TrainingLog | None = None) -> None:
        extended_to_log = {}
        if log is not None:
            extended_to_log = log.get_epoch(self._get_num_steps_from_training_state())
            if self.extended_metrics:
                epoch_time_in_seconds = extended_to_log["runtime_in_minutes"] * 60
                extended_to_log["nodes_per_second"] = self.total_num_nodes / epoch_time_in_seconds
                extended_to_log["graphs_per_second"] = (
                   self.total_num_graphs / epoch_time_in_seconds
                )

        eval_model = self.force_field
        if self.epoch_number != 0 and self.config.use_ema_params_for_eval:
            eval_model = self.ema_model
        eval_model.eval()
        eval_loss = run_evaluation(
            eval_model,
            self.eval_step,
            self.validation_dataset,
            self.epoch_number,
            self.io_handler,
            extended_to_log=extended_to_log,
        )

        if self.epoch_number == 0:
            self.best_evaluation_loss = eval_loss
            self.best_evaluation_epoch = 0

        elif eval_loss < self.best_evaluation_loss:
            logger.debug(
                "New best epoch %s has evaluation loss: %.6f",
                self.epoch_number,
                eval_loss,
            )
            self.best_evaluation_loss = eval_loss
            self.best_evaluation_epoch = self.epoch_number
            if self.config.use_ema_params_for_eval:
                self.best_model = eval_model # eval_model is already EMA-ed
            else:
                self.best_model = self.ema_model

            self.io_handler.save_checkpoint(
                (
                    flax.jax_utils.unreplicate(self.training_state.state_dict(ignore_cache=True))
                    if self.should_parallelize
                    else self.training_state.state_dict(ignore_cache=True)
                ),
                self.epoch_number,
            )

        to_log = {
            "best_loss": self.best_evaluation_loss,
            "best_epoch": self.best_evaluation_epoch,
        }
        self.io_handler.log(LogCategory.BEST_MODEL, to_log, self.epoch_number)

    def test(self, test_dataset: DataLoader) -> None:
        """Run the evaluation on the test dataset with the best parameters seen so far.

        Args:
            test_dataset: The test dataset as either a GraphDataset or
                          a PrefetchIterator.
        """
        # The following part needs to be recomputed each time as different test
        # sets could be passed in
        num_batches_test, num_graphs_test, _ = self._get_number_of_batches_graphs_and_nodes(
            test_dataset, "test"
        )
        avg_n_graphs = num_graphs_test / num_batches_test
        test_eval_step = make_evaluation_step(
            self._loss_eval,
            avg_n_graphs,
            self.should_parallelize,
        )

        self.best_model.eval()
        run_evaluation(
            self.best_model,
            test_eval_step,
            test_dataset,
            self.epoch_number,
            self.io_handler,
            is_test_set=True,
        )


    def _prepare_training_state(self, optimizer: optax.GradientTransformation) -> None:
        start_time = time.perf_counter()

        self.optimizer = nnx.Optimizer(self.force_field, optimizer, wrt=nnx.Param)
        self.ema_tracker = EMATracker(
            self.force_field, self.config.ema_decay
        )

        training_state = TrainingState(
            predictor=self.force_field,
            optimizer=self.optimizer,
            ema_tracker=self.ema_tracker,
            num_steps=TrainingStateVar(jnp.array(0)),
            acc_steps=TrainingStateVar(jnp.array(0)),
        )

        # The following line only restores the training state if the associated
        # setting in self.io_handler is set to true.
        training_state = self.io_handler.restore_training_state(training_state)

        # jax.device_put cannot handle nnx.Module directly.
        training_state_dict = jax.device_put(training_state.state_dict())
        nnx.update(training_state, training_state_dict)

        logger.debug(
            "Prepared training state on CPU in %.2f sec.",
            time.perf_counter() - start_time,
        )
        logger.info("Number of parameters: %s", _count_parameters(self.force_field.force_model))
        logger.info(
            "Number of parameters in optimizer: %s", _count_parameters(self.optimizer)
        )

        # Note: DISABLED AS IT'S MEMORY INTENSIVE AND BUT LEFT FOR VALIDATION PURPOSES.
        # assert_pytrees_match_across_hosts(training_state)
        # logger.debug(f"Training state is identical across all workers.")

        if self.should_parallelize:
            # Distribute training state
            start_time = time.perf_counter()
            training_state_dict = flax.jax_utils.replicate(training_state.state_dict())
            nnx.update(training_state, training_state_dict)
            logger.debug(
                "Distributed training state in %.2f sec.",
                time.perf_counter() - start_time,
            )

        self.training_state = training_state

    def _get_epoch_number_from_training_state(self) -> int:
        return self._get_num_steps_from_training_state() // self.steps_per_epoch

    def _get_num_steps_from_training_state(self) -> int:
        if self.should_parallelize:
            return int(self.training_state.num_steps.value[0].squeeze().block_until_ready())
        return int(self.training_state.num_steps.value.squeeze().block_until_ready())

    def _log_after_training_steps(
        self,
        metrics: dict[str, np.ndarray],
        epoch_number: int,
        num_steps: int,
    ) -> None:
        try:
            if self.extended_metrics:
                opt_hyperparams = jax.device_get(
                    nnx.pure(self.training_state.optimizer.opt_state.hyperparams)
                )
                if self.should_parallelize:
                    opt_hyperparams = flax.jax_utils.unreplicate(opt_hyperparams)
                metrics["learning_rate"] = float(opt_hyperparams["lr"])
        except AttributeError:
            pass

        metrics = convert_mse_to_rmse_in_logs(metrics)
        self.io_handler.log(LogCategory.TRAIN_METRICS, metrics, epoch_number, num_steps)

        logger.debug(
            "Total number of steps after epoch %s: %s",
            epoch_number,
            self._get_num_steps_from_training_state(),
        )

    def _get_number_of_batches_graphs_and_nodes(
        self, dataset: DataLoader, name: str
    ) -> tuple[int, int, int]:
        if isinstance(self.config.warmup_scan, dict):
            result = self.config.warmup_scan.get(name, None)
            if result is not None:
                return result
        else:
            if not self.config.warmup_scan:
                num_graphs = len(dataset.dataset)
                avg_num_nodes = self.force_field.dataset_info.avg_num_nodes
                return num_graphs // dataset.batch_size, num_graphs, num_graphs * avg_num_nodes

        total_num_graphs = 0
        total_num_nodes = 0
        total_num_batches = 0

        def _batch_generator():
            if self.should_parallelize:
                for stacked_batch in dataset:
                    for i in range(stacked_batch.n_node.shape[0]):
                        yield jax.tree.map(lambda x, idx=i: x[idx], stacked_batch)
            else:
                yield from dataset

        approx_total = len(dataset.dataset) // dataset.batch_size
        for _batch in tqdm(
            _batch_generator(), total=approx_total, desc=f"Warmup scan of {name} dataset"
        ):
            total_num_graphs += jraph.get_graph_padding_mask(_batch).sum()
            total_num_nodes += jraph.get_node_padding_mask(_batch).sum()
            total_num_batches += 1

        logger.info(
            "Warmup scan of %s dataset done. Total number of batches, graphs, and nodes: "
            "[%s, %s, %s].",
            name, total_num_batches, total_num_graphs, total_num_nodes,
        )

        return total_num_batches, total_num_graphs, total_num_nodes

    @property
    def ema_model(self) -> ForceFieldPredictor:
        """Returns the EMA-ed model."""
        _, other_state = nnx.state(self.force_field, nnx.Param, ...)
        return self.ema_tracker.get_model(other_state, parallel=self.should_parallelize)
