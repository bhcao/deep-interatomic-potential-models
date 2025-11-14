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

from typing import Any, TYPE_CHECKING

from dipm.training.training_io_handler import LogCategory
from dipm.utils._lazy_loader import LazyLoader

tensorboardX = LazyLoader(
    "tensorboardx", globals(), "tensorboardX",
    error_msg="`tensorboardX` is not installed but required for TensorBoardLogger."
)
wandb = LazyLoader(
    "wandb", globals(), "wandb",
    error_msg="`wandb` is not installed but required for WandbLogger."
)

if TYPE_CHECKING:
    import tensorboardX
    import wandb


class TensorBoardLogger:
    """Logging function for the training loop which logs the metrics to TensorBoard.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        epoch: The current epoch number used for validation and testing metrics.
        step: The current step number used for trainining metrics.
    """
    def __init__(self, logdir: str, writer: Any | None = None):
        if writer is None:
            self.writer = tensorboardX.SummaryWriter(logdir)
        else:
            self.writer = writer

        self.prefix_map = {
            LogCategory.TRAIN_METRICS: "train",
            LogCategory.EVAL_METRICS: "val",
            LogCategory.TEST_METRICS: "test",
        }

    def __call__(
        self, category: LogCategory, to_log: dict[str, Any], epoch: int, step: int
    ) -> None:
        if category not in self.prefix_map:
            return

        prefix = self.prefix_map[category]

        if category == LogCategory.TRAIN_METRICS:
            for k, v in to_log.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)
        else:
            for k, v in to_log.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, epoch)

        self.writer.flush()


class WandbLogger:
    """Logging function for the training loop which logs the metrics to Weights and Biases.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        epoch: The current epoch number used for validation and testing metrics.
        step: The current step number used for trainining metrics.
    """
    def __init__(self, project: str, run: Any | None = None):
        if run is None:
            self.run = wandb.init(project=project, reinit=True)
        else:
            self.run = run

        self.prefix_map = {
            LogCategory.TRAIN_METRICS: "train",
            LogCategory.EVAL_METRICS: "val",
            LogCategory.TEST_METRICS: "test",
        }

    def __call__(
        self, category: LogCategory, to_log: dict[str, Any], epoch: int, step: int
    ) -> None:
        if category not in self.prefix_map:
            return

        prefix = self.prefix_map[category]

        log_dict = {
            f"{prefix}/{k}": v for k, v in to_log.items()
        }

        if category == LogCategory.TRAIN_METRICS:
            self.run.log(log_dict, step=step)
        else:
            self.run.log(log_dict, step=epoch)
