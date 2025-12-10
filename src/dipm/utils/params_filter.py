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

'''Functions to filter the parameters of a model before creating it.'''

from collections.abc import Callable
import logging

import jax
import jax.numpy as jnp

from dipm.data.dataset_info import DatasetInfo
from dipm.models.force_model import ForceModelConfig, ForceModel

ParamsFilter = Callable[
    [DatasetInfo, ForceModelConfig, dict[str, jax.Array], type[ForceModel]],
    tuple[DatasetInfo, ForceModelConfig, dict[str, jax.Array]]
]

logger = logging.getLogger('dipm')


class ForceHeadParamsFilter:
    """Removes the force head if exists."""

    def __call__(
        self,
        dataset_info: DatasetInfo,
        model_config: ForceModelConfig,
        params: dict[str, jax.Array],
        model_type: type[ForceModel],
    ) -> tuple[DatasetInfo, ForceModelConfig, dict[str, jax.Array]]:
        if model_config.force_head:
            params = {
                k: v
                for k, v in params.items()
                if not k.startswith(model_type.force_head_prefix)
            }
            model_config.force_head = False
            logger.info("Force head has been dropped from the loaded model.")
        return dataset_info, model_config, params


class UnseenElementsParamsFilter:
    """Removes the parameters of unseen elements."""

    def __init__(self, elements_to_drop: set[int]):
        self.elements_to_drop = elements_to_drop

    def __call__(
        self,
        dataset_info: DatasetInfo,
        model_config: ForceModelConfig,
        params: dict[str, jax.Array],
        model_type: type[ForceModel],
    ) -> tuple[DatasetInfo, ForceModelConfig, dict[str, jax.Array]]:
        atomic_energies_map = {}
        index_to_keep = []
        for i, k in enumerate(sorted(dataset_info.atomic_energies_map)):
            if k in self.elements_to_drop:
                continue
            atomic_energies_map[k] = dataset_info.atomic_energies_map[k]
            index_to_keep.append(i)
        index_to_keep = jnp.array(index_to_keep)

        # update parameters
        original_num_species = len(dataset_info.atomic_energies_map)
        for k in list(params.keys()):
            if model_type.embedding_layer_regexp.search(k):
                if len(params[k]) != original_num_species:
                    raise ValueError(
                        f"Shape of parameter {k} is mismatched. Expected {original_num_species},"
                        f" got {len(params[k])}."
                    )
                params[k] = params[k][index_to_keep]
        dataset_info.atomic_energies_map = atomic_energies_map

        return dataset_info, model_config, params
