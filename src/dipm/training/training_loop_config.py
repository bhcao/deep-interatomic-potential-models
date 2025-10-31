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
from typing_extensions import Annotated

PositiveInt = Annotated[int, pydantic.Field(gt=0)]
EMADecay = Annotated[float, pydantic.Field(gt=0.0, le=1.0)]


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
        log_per_steps: Number of steps to log the metrics. Default to 10000.
    """

    num_epochs: PositiveInt
    num_gradient_accumulation_steps: PositiveInt = 1
    ema_decay: EMADecay = 0.99
    use_ema_params_for_eval: bool = True
    run_eval_at_start: bool = True
    log_per_steps: PositiveInt = 10000
