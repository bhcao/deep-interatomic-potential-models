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
from copy import deepcopy
from typing import Any

from rich import print as table_print
from rich.table import Table

from dipm.training.training_io_handler import LogCategory


class TableLogger:
    """Logging function for the training loop which logs the metrics to a nice table.

    The table will be printed to the command line.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        epoch: The current epoch number used for validation and testing metrics.
        step: The current step number used for trainining metrics.
    """
    @staticmethod
    def _build_rich_table(
        metrics: dict[str, int | float], title: str, metric_color: str
    ) -> Table:
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        keys = [k.replace("_", " ").capitalize() for k in metrics.keys()]
        values = [v if isinstance(v, int) else f"{float(v):.3f}" for v in metrics.values()]

        for k, v in zip(keys, values):
            _k = f"[{metric_color}]{k}[/{metric_color}]"
            _v = f"[white]{v}[/white]"
            table.add_row(_k, _v)

        return table

    def __call__(
        self, category: LogCategory, to_log: dict[str, Any], epoch: int, step: int
    ) -> None:
        table = None
        if category == LogCategory.BEST_MODEL:
            table = TableLogger._build_rich_table(to_log, f"Best model at epoch {epoch}", "white")
        elif category == LogCategory.TRAIN_METRICS:
            table = TableLogger._build_rich_table(
                to_log, f"Training set metrics at step {step}", "cyan"
            )
        elif category == LogCategory.EVAL_METRICS:
            table = TableLogger._build_rich_table(
                to_log, f"Validation set metrics at epoch {epoch}", "green"
            )
        elif category == LogCategory.TEST_METRICS:
            table = TableLogger._build_rich_table(
                to_log, f"Test set metrics at epoch {epoch}", "blue_violet"
            )

        if table is not None:
            table_print(table)


class LineLogger:
    """Logging function for the training loop which logs the metrics to a single line.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        epoch: The current epoch number used for validation and testing metrics.
        step: The current step number used for trainining metrics.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def _metrics_to_str(metrics: dict[str, Any]) -> str:
        result = ""
        for k, v in metrics.items():
            result += k.replace("_", " ").capitalize()
            result += " = "
            result += str(v if isinstance(v, int) else f"{float(v):.3f}")
            result += " | "

        if result == "":
            return result
        return result[:-3]  # remove final separator

    def __call__(
        self, category: LogCategory, to_log: dict[str, Any], epoch: int, step: int
    ) -> None:
        if category == LogCategory.BEST_MODEL:
            self.logger.info(
                "Best model: Loss = %.3f | Best epoch = %s",
                to_log["best_loss"],
                to_log["best_epoch"],
            )

        elif category == LogCategory.TRAIN_METRICS:
            # Make sure loss is first
            to_log_copy = deepcopy(to_log)
            to_log_modified = {"loss": to_log_copy.pop("loss")}
            to_log_modified.update(to_log_copy)
            to_log_modified["steps_in_total"] = step

            self.logger.info("%-11s %s", "Training:", LineLogger._metrics_to_str(to_log_modified))

        elif category == LogCategory.EPOCH_START:
            self.logger.info("------------ Epoch %s ------------", epoch)

        elif category == LogCategory.EVAL_METRICS:
            self.logger.info("Validation: %s", LineLogger._metrics_to_str(to_log))

        elif category == LogCategory.TEST_METRICS:
            self.logger.info("Testing: %s", LineLogger._metrics_to_str(to_log))
