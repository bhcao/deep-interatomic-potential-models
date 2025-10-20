# Copyright 2025 Cao Bohan
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

"""A LazyLoader class borrowed from TensorFlow."""

import importlib
import logging
import types

logger = logging.getLogger('dipm-cvt-cli')


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
                logger.critical(self._error_msg)
            else:
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
