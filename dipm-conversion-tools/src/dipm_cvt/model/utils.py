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

from collections import defaultdict
from collections.abc import Callable
import logging
import re

import numpy as np

logger = logging.getLogger('dipm-cvt-model')


class _Rule:
    '''Map state_dict keys from old format to new format.'''

    def __init__(
        self,
        patterns: list[str],
        repl: str | None = None,
        transform_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None
    ):
        self.patterns = [re.compile(pattern) for pattern in patterns]
        self.repl = repl
        if len(self.patterns) == 0:
            raise ValueError('TransformRule must have at least one pattern')
        if len(self.patterns) == 1:
            self.transform_fn = transform_fn or (lambda x: x[0])
        else: # Such as to_k, to_v, to_q -> to_kvq
            self.transform_fn = transform_fn or (lambda x: np.concatenate(x, axis=-1))

    def apply(self, state_dict: dict[str, np.ndarray]):
        '''Apply the transformation rule to the state dict.
        
        Args:
            state_dict: The state dict to be transformed.
            new_state_dict: The transformed state dict.
        '''

        # Delete keys
        if self.repl is None:
            for key in list(state_dict.keys()):
                if any(p.search(key) for p in self.patterns):
                    del state_dict[key]
            return

        # Rename and merge
        match_groups = defaultdict(list)
        for pattern in self.patterns:
            for key in list(state_dict.keys()):
                if pattern.search(key):
                    new_key = pattern.sub(self.repl, key)
                    match_groups[new_key].append(state_dict[key])
                    del state_dict[key]

        for new_key, values in match_groups.items():
            if len(values) != len(self.patterns):
                logger.critical('New state key %s only matches %d values', new_key, len(values))
            state_dict[new_key] = self.transform_fn(values)

        return


class StateDictConverter:
    '''Convert state dict from old format to new format.'''

    def __init__(self, rules_dict: dict):
        self.rules = []
        for key, value in rules_dict.items():
            if isinstance(key, str):
                key = [key]
            if not isinstance(value, tuple):
                value = (value,)
            self.rules.append(_Rule(key, *value))

    def apply(self, state_dict: dict[str, np.ndarray]):
        '''Convert the state dict.'''

        for rule in self.rules:
            rule.apply(state_dict)
