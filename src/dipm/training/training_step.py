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

from dataclasses import field
import functools
from collections.abc import Callable

import jax
import optax
from jax import Array
from flax import nnx
from flax import struct
from jraph import GraphsTuple

from dipm.training.metrics_reweighting import reweight_metrics_by_number_of_graphs
from dipm.typing import LossFunction
from dipm.models import ForceFieldPredictor
from dipm.training.optimizer import EMATracker


@struct.dataclass
class TrainingState:
    """
    Represents the state of training.

    Attributes:
        model: ForceField Model.
        optimizer: NNX optimizer.
        ema_tracker: Exponentially weighted average tracker.
        num_steps: The number of training steps taken.
        acc_steps: The number of gradient accumulation steps taken; resets to 0 after
                   each optimizer step.
        extras: Additional auxiliary information in form of a dictionary.
    """

    predictor: ForceFieldPredictor
    optimizer: nnx.Optimizer
    ema_tracker: EMATracker
    num_steps: jax.Array
    acc_steps: jax.Array
    extras: dict | None = field(default_factory=dict) # pylint: disable=invalid-field-call

    def state_dict(self) -> dict:
        '''Return dict of nnx.State (filter out RNGs)'''
        wrt = [nnx.Param, nnx.BatchStat]
        return {
            "predictor": nnx.state(self.predictor, wrt),
            "optimizer": nnx.state(self.optimizer),
            "ema_tracker": nnx.state(self.ema_tracker),
            "num_steps": self.num_steps,
            "acc_steps": self.acc_steps,
            "extras": self.extras,
        }

    def load_state_dict(self, state_dict: dict, model_only: bool = False):
        '''Restore from dict of nnx.State'''
        nnx.update(self.predictor, state_dict["predictor"])
        if not model_only:
            nnx.update(self.optimizer, state_dict["optimizer"])
            nnx.update(self.ema_tracker, state_dict["ema_tracker"])
            # pylint: disable=no-member
            self.replace(
                num_steps = state_dict["num_steps"],
                acc_steps = state_dict["acc_steps"],
                extras = state_dict["extras"],
            )


def _training_step(
    training_state: TrainingState,
    graph: GraphsTuple,
    rngs: nnx.Rngs,
    epoch_number: int,
    loss_fun: LossFunction,
    avg_n_graphs_per_batch: float,
    num_gradient_accumulation_steps: int | None,
    should_parallelize: bool,
) -> dict:
    '''Training state will be updated rather than create a new state.'''

    # Fetch params and optimizer state from training state.
    predictor = training_state.predictor
    optimizer = training_state.optimizer
    ema_tracker = training_state.ema_tracker
    num_steps = training_state.num_steps
    acc_steps = training_state.acc_steps

    def model_loss_fun(
        predictor: nnx.Module, ref_graph: GraphsTuple, rngs: nnx.Rngs, epoch: int
    ) -> tuple[Array, dict]:
        predictions = predictor(ref_graph, rngs)
        return loss_fun(predictions, ref_graph, epoch)

    # Calculate gradients.
    grad_fun = nnx.grad(model_loss_fun, argnums=0, has_aux=True)
    grads, aux_info = grad_fun(predictor, graph, rngs, epoch_number)

    # Aggregrate over devices.
    if should_parallelize:
        grads = jax.lax.pmean(grads, axis_name="device")

    # Gradient step on params.
    optimizer.update(predictor, grads)
    ema_tracker.update(predictor)

    # Fetch logging info from aux_info.
    metrics = aux_info

    # Reweight metrics to account for different number of real graphs per batch
    metrics = reweight_metrics_by_number_of_graphs(
        metrics, graph, avg_n_graphs_per_batch
    )

    # Add batch-level metrics to the dictionary.
    metrics["gradient_norm"] = optax.global_norm(grads)
    # NOTE: Since it's hard to get updates in nnx, `param_update_norm` is ignored

    # Aggregate over global devices.
    if should_parallelize:
        metrics = jax.lax.pmean(metrics, axis_name="device")

    # Update per-step variables.
    training_state.replace(
        acc_steps = (acc_steps + 1) % num_gradient_accumulation_steps,
        num_steps = jax.lax.cond(acc_steps == 0, lambda x: x + 1, lambda x: x, num_steps),
    )

    return metrics


def make_train_step(
    loss_fun: LossFunction,
    avg_n_graphs_per_batch: float,
    num_gradient_accumulation_steps: int | None = 1,
    should_parallelize: bool = True,
) -> Callable[[TrainingState, GraphsTuple, nnx.Rngs, int], dict]:
    """
    Create a training step function to optimize model params using gradients.

    Args:
        loss_fun: A function that computes the loss from predictions, a reference
                  labelled graph, and the epoch number.
        avg_n_graphs_per_batch: Average number of graphs per batch used for
                                reweighting of metrics.
        num_gradient_accumulation_steps: The number of gradient accumulation
                                         steps before a parameter update is performed.
                                         Defaults to 1, implying immediate updates.
        should_parallelize: Whether to apply pmap.

    Returns:
        A function that takes the current training state and a batch of data as
        input, and returns the updated training state along with training metrics.
    """

    training_step = functools.partial(
        _training_step,
        loss_fun=loss_fun,
        avg_n_graphs_per_batch=avg_n_graphs_per_batch,
        num_gradient_accumulation_steps=num_gradient_accumulation_steps,
        should_parallelize=should_parallelize,
    )

    if should_parallelize:
        return jax.pmap(
            nnx.jit(training_step),
            axis_name="device",
            static_broadcasted_argnums=2,
        )
    return nnx.jit(training_step)
