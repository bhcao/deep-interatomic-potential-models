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

import pydantic
from pydantic import Field, model_validator
from typing_extensions import Self, Annotated

from dipm.simulation.enums import SimulationType, TemperatureScheduleMethod
from dipm.typing import PositiveFloat, PositiveInt

logger = logging.getLogger("dipm")

NUM_STEPS_LOGGING_THRESHOLD = 1_000
MIN_LOG_FREQ = 10
MAX_LOG_FREQ = 1_000
DEFAULT_EDGE_CAPACITY_MULT = 1.25
NUM_STEPS_PER_EP_THRESHOLD = 1_000
MIN_STEPS_PER_EP = 10
MAX_STEPS_PER_EP = 1_000

ThreeDimensionalListWithPositiveFloats = Annotated[
    list[PositiveFloat], pydantic.Field(min_length=3, max_length=3)
]
FloatLargerThanOrEqualToOne = Annotated[float, pydantic.Field(ge=1)]


class TemperatureScheduleConfig(pydantic.BaseModel):
    """The base configuration containing all the possible parameters for the
    temperature schedules.

    Attributes:
        method: The type of temperature schedule to use. Default is constant.
        temperature: The temperature to use for the constant schedule in Kelvin.
        start_temperature: The starting temperature in Kelvin.
            Used for the linear schedule.
        end_temperature: The ending temperature in Kelvin.
            Used for the linear schedule.
        max_temperature: The maximum temperature in Kelvin.
            Used for the triangle schedule.
        min_temperature: The minimum temperature in Kelvin.
            Used for the triangle schedule.
        heating_period: The period for heating the system.
            Measured in number of simulation steps. Used for the triangle schedule.

    """

    method: TemperatureScheduleMethod = Field(
        default=TemperatureScheduleMethod.CONSTANT
    )

    # Constant schedule
    temperature: PositiveFloat | None = None

    # Linear schedule
    start_temperature: PositiveFloat | None = None
    end_temperature: PositiveFloat | None = None

    # Triangle schedule
    max_temperature: PositiveFloat | None = None
    min_temperature: PositiveFloat | None = None
    heating_period: PositiveInt | None = None


class SimulationConfig(pydantic.BaseModel):
    """The base configuration that all simulations share.

    It only contains fields that are independent of backend and simulation type.

    Attributes:
        simulation_type: The type of simulation to run, either MD or minimization.
                    Defaults to MD.
        num_steps: The number of total steps to run. For energy minimizations,
                   this is the maximum number of steps if no convergence reached
                   earlier.
        snapshot_interval: The interval (in steps) between snapshots of the simulation
                           state. This means information about every N-th
                           snapshot is stored in the simulation state available to the
                           loggers (N being the snapshot interval). Defaults to 1.
        box: The simulation box. If ``None``, no periodic boundary conditions are
             applied (this is the default). It can be set to either a float or a list
             of three floats, describing the dimensions of the box.
        edge_capacity_multiplier: Factor to multiply the number of edges by to
                                  obtain the edge capacity including padding. Defaults
                                  to 1.25.
    """

    simulation_type: SimulationType = SimulationType.MD
    num_steps: PositiveInt
    snapshot_interval: PositiveInt = 1
    box: PositiveFloat | ThreeDimensionalListWithPositiveFloats | None = None
    edge_capacity_multiplier: FloatLargerThanOrEqualToOne = DEFAULT_EDGE_CAPACITY_MULT


class JaxMDSimulationConfig(SimulationConfig):
    """Configuration for the JAX-MD-based simulations.

    The config is separated into three blocks: values that
    are used for both MD and minimization, and then the ones used exclusively for MD
    and for minimization, respectively.

    Attributes:
        num_episodes: Number of episodes to divide the simulation into. Each episode
                      runs in a fully jitted way, and the loggers are only
                      called after each episode. If not set, an appropriate value will
                      be attempted to be select but it is possible that it may have to
                      be manually set. For fewer than 1000 steps, ``num_episodes`` will
                      be set so that the number of steps per episode will be 10. For
                      more than 1000 steps, ``num_episodes`` will be set so that the
                      number of steps per episode will be 1000. Therefore, if
                      ``num_episodes`` is not set, it requires that ``num_steps`` be
                      divisible by 1000 if greater than 1000 otherwise divisible by 10.
        timestep_fs: The simulation timestep in femtoseconds. This is also used as the
                     initial timestep in the FIRE minimization algorithm. The default is
                     1.0. **Important Note:** We recommend to set this value to 0.1 when
                     running energy minimizations instead of MD simulations.
        temperature_kelvin: The temperature in Kelvin, set to 300 by default. Must be
                            set to ``None`` for energy minimizations.
        temperature_schedule_config: The temperature schedule config to use for the
                                 simulation. Default is the constant schedule in
                                 which case ``temperature_kelvin`` will be applied.
    """

    num_episodes: PositiveInt | None = None
    timestep_fs: PositiveFloat | None = 1.0

    # MD only
    temperature_kelvin: PositiveFloat | None = 300.0
    temperature_schedule_config: TemperatureScheduleConfig = Field(
        default=TemperatureScheduleConfig(temperature=temperature_kelvin)
    )

    @model_validator(mode="after")
    def validate_num_episodes(self) -> Self:
        if not self.num_episodes:
            if self.num_steps < NUM_STEPS_PER_EP_THRESHOLD:
                self.num_episodes = max(self.num_steps // MIN_STEPS_PER_EP, 1)
            else:
                self.num_episodes = self.num_steps // MAX_STEPS_PER_EP
        if self.num_steps % self.num_episodes > 0:
            raise ValueError("Number of episodes must evenly divide total steps.")
        return self

    @model_validator(mode="after")
    def validate_snapshot_interval(self) -> Self:
        steps_per_episode = self.num_steps // self.num_episodes
        if steps_per_episode % self.snapshot_interval > 0:
            raise ValueError("Snapshot interval must evenly divide steps per episode.")
        return self

class ASESimulationConfig(SimulationConfig):
    """Configuration for the ASE-based simulations.

    Also includes the attributes of the parent class
    :class:`~mlip.simulation.configs.simulation_config.SimulationConfig`.

    The config is separated into three blocks: values that
    are used for both MD and minimization, and then the ones used exclusively for MD
    and for minimization, respectively.

    Attributes:
        log_interval: The interval in ``num_steps`` at which the loggers
                      will be called. If not set, an appropriate value will
                      be attempted to be selected. For fewer than 1000 steps,
                      it will default to 10. For more than 1000 steps, it will
                      default to 1000.
        timestep_fs: The simulation timestep in femtoseconds. The default is
                     1.0.
        temperature_kelvin: The temperature in Kelvin, set to 300 by default. Must be
                            set to ``None`` for energy minimizations.
        friction: Friction coefficient for the simulation. Default is 0.1.
        temperature_schedule_config: The temperature schedule config to use for the
                                   simulation. Default is the constant schedule in
                                   which case ``temperature_kelvin`` will be applied.
        max_force_convergence_threshold: The convergence threshold for minimizations
                                         w.r.t. the sum of the force norms. See the
                                         ASE docs for more information. If not set,
                                         the ASE default will be used.
    """

    log_interval: PositiveInt | None = None

    # MD Only
    timestep_fs: PositiveFloat | None = 1.0
    temperature_kelvin: PositiveFloat | None = 300.0
    friction: PositiveFloat | None = 0.1

    # Temperature scheduling for MD
    temperature_schedule_config: TemperatureScheduleConfig = Field(
        default=TemperatureScheduleConfig(temperature=temperature_kelvin)
    )

    # Minimization only
    max_force_convergence_threshold: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_log_interval(self) -> Self:
        if not self.log_interval:
            if self.num_steps < NUM_STEPS_LOGGING_THRESHOLD:
                self.log_interval = MIN_LOG_FREQ
            else:
                self.log_interval = MAX_LOG_FREQ
        return self

    @model_validator(mode="after")
    def validate_snapshot_interval(self) -> Self:
        if self.num_steps % self.snapshot_interval > 0:
            logger.warning(
                "It is best for snapshot interval to divide"
                " the number of steps otherwise the final state of"
                " the simulation will not be saved.",
            )
        return self
