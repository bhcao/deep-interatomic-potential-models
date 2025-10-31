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

from dipm.data.helpers.data_prefetching import PrefetchIterator
from dipm.data.helpers.graph_dataset import GraphDataset
from dipm.models import ForceFieldPredictor
from dipm.loss.loss import Loss
from dipm.training.optimizer import EMATracker
from dipm.training.evaluation import make_evaluation_step, run_evaluation
from dipm.training.training_io_handler import LogCategory, TrainingIOHandler
from dipm.training.training_loggers import log_metrics_to_line
from dipm.training.training_loop_config import TrainingLoopConfig
from dipm.training.training_step import TrainingState, make_train_step

GraphDatasetOrPrefetchIterator: TypeAlias = GraphDataset | PrefetchIterator
TrainingStepFun: TypeAlias = Callable[
    [TrainingState, jraph.GraphsTuple],
    tuple[TrainingState, dict],
]

logger = logging.getLogger("dipm")


def _count_parameters(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))


class _TrainingLog:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.last_log_step = 0
        self.metrics: list[dict] = []
        self.start_time = time.perf_counter()

    def append(self, metrics: dict) -> None:
        '''Appends metrics of a training.'''
        self.metrics.append(metrics)
        self.num_steps += 1

    def get(self):
        '''Returns the metrics of the training.'''
        end_time = time.perf_counter()
        steps = self.num_steps - self.last_log_step
        time_per_step = (end_time - self.start_time) / steps
        metrics = {}
        for metric_name in self.metrics[0].keys():
            metrics[metric_name] = np.mean([m[metric_name] for m in self.metrics])
        # Update
        self.last_log_step = self.num_steps
        self.start_time = end_time
        self.metrics = []
        return metrics, time_per_step


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
        train_dataset: GraphDatasetOrPrefetchIterator,
        validation_dataset: GraphDatasetOrPrefetchIterator,
        force_field: ForceFieldPredictor,
        loss: Loss,
        optimizer: optax.GradientTransformation,
        config: TrainingLoopConfig,
        io_handler: TrainingIOHandler | None = None,
        should_parallelize: bool = False,
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
            should_parallelize: Whether to parallelize (using data parallelization)
                                across multiple devices. The default is ``False``.
        """
        self.should_parallelize = should_parallelize

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.total_num_graphs, self.total_num_nodes = (
            self._get_total_number_of_graphs_and_nodes_in_dataset(self.train_dataset)
        )

        self.force_field = force_field
        self.config = config

        self.extended_metrics = (
            True if not hasattr(loss, "extended_metrics") else loss.extended_metrics
        )
        self.io_handler = io_handler
        if self.io_handler is None:
            self.io_handler = TrainingIOHandler()
            self.io_handler.attach_logger(log_metrics_to_line)

        self.io_handler.save_dataset_info(self.force_field.dataset_info)

        self._loss_train = partial(loss, eval_metrics=False)
        self._loss_eval = partial(loss, eval_metrics=True)

        self._prepare_training_state(optimizer)
        # Note: Because we shuffle the training data between epochs, the following
        # value may slightly fluctuate during training, however, we assume
        # it being fixed, which is a solid approximation for datasets of typical size.
        _avg_n_graphs_train = self.total_num_graphs / len(self.train_dataset)
        self.training_step = make_train_step(
            self._loss_train,
            _avg_n_graphs_train,
            config.num_gradient_accumulation_steps,
            should_parallelize,
        )
        self.metrics = None
        _avg_n_graphs_validation = (
            self._get_total_number_of_graphs_and_nodes_in_dataset(
                self.validation_dataset
            )[0]
            / len(self.validation_dataset)
        )
        # Clone of force_field, will be updated every training round.
        self.best_model = nnx.clone(force_field)
        self.eval_step = make_evaluation_step(
            self._loss_eval,
            _avg_n_graphs_validation,
            should_parallelize,
        )

        self.best_evaluation_step = -1
        self.best_evaluation_loss = float("inf")
        self.best_evaluation_epoch = -1

        self._should_unreplicate_train_batches = (
            not should_parallelize
        ) and isinstance(self.train_dataset, PrefetchIterator)

        self.num_batches = len(self.train_dataset)
        self.steps_per_epoch = self.num_batches
        if should_parallelize:
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

    def run(self) -> None:
        """Runs the training loop.

        The final training state can be accessed via its member variable.
        """
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

        train_rngs = nnx.Rngs(42)

        log = _TrainingLog(self.epoch_number * self.steps_per_epoch)
        while self.epoch_number < self.config.num_epochs:
            self.epoch_number += 1
            self.io_handler.log(LogCategory.EPOCH_START, {}, self.epoch_number)
            t_before_train = time.perf_counter()
            self._run_training_epoch(train_rngs, log)
            logger.debug(
                "Parameter updates of epoch %s done, running evaluation next.",
                self.epoch_number,
            )
            t_after_train = time.perf_counter()
            self._run_evaluation()
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

        for batch in self.train_dataset:
            if self._should_unreplicate_train_batches:
                batch = flax.jax_utils.unreplicate(batch)
            _metrics = self.training_step(
                self.training_state, batch, rngs, self.epoch_number
            )
            log.append(jax.device_get(_metrics))

            if log.num_steps % self.config.log_per_steps == 0:
                metrics, time_per_step = log.get()
                self._log_after_training_steps(
                    metrics, self.epoch_number, time_per_step, log.num_steps
                )

    def _run_evaluation(self) -> None:
        devices = jax.devices() if self.should_parallelize else None

        eval_model = self.force_field
        if self.epoch_number != 0 and self.config.use_ema_params_for_eval:
            eval_model = self._get_model_from_training_state()
        eval_model.eval()
        eval_loss = run_evaluation(
            eval_model,
            self.eval_step,
            self.validation_dataset,
            self.epoch_number,
            self.io_handler,
            devices,
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
                self.best_model = self._get_model_from_training_state()

            self.io_handler.save_checkpoint(
                (
                    flax.jax_utils.unreplicate(self.training_state)
                    if self.should_parallelize
                    else self.training_state
                ),
                self.epoch_number,
            )

        to_log = {
            "best_loss": self.best_evaluation_loss,
            "best_epoch": self.best_evaluation_epoch,
        }
        self.io_handler.log(LogCategory.BEST_MODEL, to_log, self.epoch_number)

    def test(self, test_dataset: GraphDatasetOrPrefetchIterator) -> None:
        """Run the evaluation on the test dataset with the best parameters seen so far.

        Args:
            test_dataset: The test dataset as either a GraphDataset or
                          a PrefetchIterator.
        """
        devices = jax.devices() if self.should_parallelize else None

        # The following part needs to be recomputed each time as different test
        # sets could be passed in
        avg_n_graphs = self._get_total_number_of_graphs_and_nodes_in_dataset(
            test_dataset
        )[0] / len(test_dataset)
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
            devices,
            is_test_set=True,
        )


    def _prepare_training_state(self, optimizer: optax.GradientTransformation) -> None:
        start_time = time.perf_counter()

        self.optimizer = nnx.Optimizer(
            self.force_field, optimizer, wrt=[nnx.Param, nnx.BatchStat]
        )
        self.ema_tracker = EMATracker(
            self.force_field, self.config.ema_decay
        )

        training_state = TrainingState(
            predictor=self.force_field,
            optimizer=self.optimizer,
            ema_tracker=self.ema_tracker,
            num_steps=jnp.array(0),
            acc_steps=jnp.array(0),
            extras={},
        )

        # The following line only restores the training state if the associated
        # setting in self.io_handler is set to true.
        training_state = self.io_handler.restore_training_state(training_state)

        # jax.device_put cannot handle nnx.Module directly.
        training_state_dict = jax.device_put(training_state.state_dict())
        training_state.load_state_dict(training_state_dict)

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
            training_state = flax.jax_utils.replicate(training_state)
            logger.debug(
                "Distributed training state in %.2f sec.",
                time.perf_counter() - start_time,
            )

            # Distribute keys
            start_time = time.perf_counter()
            training_state.replace(acc_steps = flax.jax_utils.replicate(0))
            logger.debug(
                "Distributed keys in %.2f sec.", time.perf_counter() - start_time
            )

        self.training_state = training_state

    def _get_epoch_number_from_training_state(self) -> int:
        return self._get_num_steps_from_training_state() // self.steps_per_epoch

    def _get_num_steps_from_training_state(self) -> int:
        if self.should_parallelize:
            return int(self.training_state.num_steps[0].squeeze().block_until_ready())
        return int(self.training_state.num_steps.squeeze().block_until_ready())

    def _log_after_training_steps(
        self,
        metrics: dict[str, np.ndarray],
        epoch_number: int,
        time_per_step: float,
        num_steps: int,
    ) -> None:
        try:
            opt_hyperparams = jax.device_get(
                self.training_state.optimizer.opt_state.hyperparams
            )
            if self.should_parallelize:
                opt_hyperparams = flax.jax_utils.unreplicate(opt_hyperparams)
            if self.extended_metrics:
                metrics["learning_rate"] = float(opt_hyperparams["lr"])
        except AttributeError:
            pass

        metrics["steps_in_total"] = num_steps
        metrics["seconds_per_step"] = time_per_step
        # TODO(bhcao): Add more extended_metrics

        self.io_handler.log(LogCategory.TRAIN_METRICS, metrics, epoch_number)

        logger.debug(
            "Total number of steps after epoch %s: %s",
            epoch_number,
            self._get_num_steps_from_training_state(),
        )

    def _get_total_number_of_graphs_and_nodes_in_dataset(
        self, dataset: GraphDataset | PrefetchIterator
    ) -> tuple[int, int]:
        total_num_graphs = 0
        total_num_nodes = 0

        def _batch_generator():
            if isinstance(dataset, PrefetchIterator):
                for stacked_batch in dataset:
                    for i in range(stacked_batch.n_node.shape[0]):
                        yield jax.tree.map(lambda x, idx=i: x[idx], stacked_batch)
            else:
                yield from dataset

        for _batch in _batch_generator():
            total_num_graphs += jraph.get_graph_padding_mask(_batch).sum()
            total_num_nodes += jraph.get_node_padding_mask(_batch).sum()

        return total_num_graphs, total_num_nodes

    def _get_model_from_training_state(self) -> ForceFieldPredictor:
        devices = jax.devices() if self.should_parallelize else None

        if devices is not None:
            return jax.pmap(
                self.training_state.ema_tracker.get_model,
                axis_name="device",
                static_broadcasted_argnums=(1,),
            )()

        return self.training_state.ema_tracker.get_model()
