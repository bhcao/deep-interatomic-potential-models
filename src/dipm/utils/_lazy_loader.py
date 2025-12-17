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

"""A LazyLoader class borrowed from TensorFlow."""

import importlib
import logging
import types

logger = logging.getLogger('dipm')


class LazyLoader(types.ModuleType):
    """Lazily import a module."""

    def __init__(self, local_name, parent_module_globals, name, *,
                 error_msg=None, replace_name=None, warning_msg=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        # Error mode: raise an ImportError with the given message.
        # Warning mode: log a warning with the given message and replace the module with the given
        #   replacement module.
        in_error_mode = error_msg is not None
        in_warning_mode = warning_msg is not None and replace_name is not None
        if in_error_mode and in_warning_mode:
            raise ValueError(
                "Cannot specify both `error_msg` and `warning_msg` and `replace_name`."
            )
        if not in_error_mode and not in_warning_mode:
            raise ValueError(
                "Either `error_msg` or both `replace_name` and `warning_msg` must be specified."
            )

        self._error_mode = in_error_mode
        self._error_msg = error_msg
        self._warning_msg = warning_msg
        self._target_name = name # Replace use of self.__name__ for replaceable modules.
        self._replace_name = replace_name

        super().__init__(name)

    def _load(self):
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self._target_name)
        except ModuleNotFoundError as e:
            if e.name != self._target_name:
                raise e # Dependency of `self._target_name` is missing.
            if self._error_mode:
                raise ImportError(self._error_msg) from e
            self._target_name = self._replace_name
            logger.warning(self._warning_msg)
            module = importlib.import_module(self._replace_name)

        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
