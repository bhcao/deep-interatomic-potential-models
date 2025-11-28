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

import pydantic

from dipm.typing import (
    PositiveFloat, PositiveInt, NonNegativeFloat, NonNegativeInt, Proportion, PathLike
)


class OptimizerConfig(pydantic.BaseModel):
    """Pydantic config holding all settings that are relevant for the optimizer.

    Attributes:
        apply_weight_decay_mask: Whether to apply a weight decay mask. If set to
                                 ``False``, a weight decay is applied to all parameters.
                                 If set to ``True`` (default), only the parameters of
                                 model blocks "linear_down" and "SymmetricContraction"
                                 are assigned a weight decay. These blocks only exist
                                 for MACE models, and it is recommended for MACE to
                                 set this setting to ``True``. If it is set to
                                 ``True`` but neither of these blocks exist in the
                                 model (like for ViSNet or NequIP),
                                 we apply weight decay to all parameters.
        weight_decay: The weight decay with a default of zero.
        grad_norm: Gradient norm used for gradient clipping.
        num_gradient_accumulation_steps: Number of gradient steps to accumulate before
                                         taking an optimizer step. Default is 1.
        init_learning_rate: Initial learning rate (default is 0.01).
        peak_learning_rate: Peak learning rate (default is 0.01).
        final_learning_rate: Final learning rate (default is 0.01).
        warmup_steps: Number of optimizer warm-up steps (default is 4000).
                      Check optax's ``linear_schedule()`` function for more info.
        transition_steps: Number of optimizer transition steps (default is 360000).
                          Check optax's ``linear_schedule()`` function for more info.
    """

    apply_weight_decay_mask: bool = True

    weight_decay: NonNegativeFloat = 0.0
    grad_norm: NonNegativeFloat = 500
    num_gradient_accumulation_steps: PositiveInt = 1

    init_learning_rate: PositiveFloat = 0.01
    peak_learning_rate: PositiveFloat = 0.01
    final_learning_rate: PositiveFloat = 0.01
    warmup_steps: NonNegativeInt = 4000
    transition_steps: NonNegativeInt = 360000


class TrainingLoopConfig(pydantic.BaseModel):
    """Pydantic config holding all settings related to the
    :class:`~dipm.training.training_loop.TrainingLoop` class.

    Use `valid_num_to_load` instead of `eval_num_graphs`.

    Attributes:
        num_epochs: Number of epoch to run.
        num_gradient_accumulation_steps: Number of gradient steps to accumulate before
                                         taking an optimizer step. Default is 1.
        ema_decay: The EMA decay rate, by default set to 0.99.
        use_ema_params_for_eval: Whether to use the EMA parameters for evaluation,
                                 set to ``True`` by default.
        run_eval_at_start: Whether to run an evaluation on the validation set before
                           we start the first epoch. By default, it is set to ``True``.
        log_interval: Number of steps to log the metrics. Default is ``None``, which
                      means logging once per epoch.
    """

    num_epochs: PositiveInt
    num_gradient_accumulation_steps: PositiveInt = 1
    ema_decay: Proportion = 0.99
    use_ema_params_for_eval: bool = True
    run_eval_at_start: bool = True
    log_interval: PositiveInt | None = None


class TrainingIOHandlerConfig(pydantic.BaseModel):
    """Pydantic config holding all settings relevant for the training IO handler.

    Attributes:
        local_model_output_dir: Path to the output directory (local filesystem) where
                                the model/dataset information and checkpoints are
                                stored. If `None`, then local checkpointing will be
                                disabled. Defaults to `None`.
        max_checkpoints_to_keep: Maximum number of old checkpoints to keep.
                                 The default is 5.
        ema_decay: The EMA decay rate. The default is 0.99.
        restore_checkpoint_if_exists: Whether to restore a previous checkpoint if it
                                      exists. By default, this is ``False``.
        epoch_to_restore: The epoch number to restore. The default is ``None``, which
                          means the latest epoch will be restored.
        restore_optimizer_state: Whether to also restore the optimizer state.
                                 Default is ``False``.
        clear_previous_checkpoints: Whether to clear the previous checkpoints if
                                    any exist. Note that this setting can not be set to
                                    ``True`` if one selects to restore a checkpoint.
                                    The default is ``False``.
    """

    local_model_output_dir: PathLike | None = None
    max_checkpoints_to_keep: PositiveInt = 5
    ema_decay: Proportion = 0.99

    restore_checkpoint_if_exists: bool = False
    epoch_to_restore: PositiveInt | None = None
    restore_optimizer_state: bool = False
    clear_previous_checkpoints: bool = False
