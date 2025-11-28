# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

from e3nn_jax import IrrepsArray
from flax.typing import Dtype
from jax import numpy as jnp

T = typing.TypeVar('T', bound=tuple)


def canonicalize_dtype(
    *args, dtype: Dtype | None = None, inexact: bool = True
) -> Dtype:
    """Canonicalize an optional dtype to the definitive dtype.

    Modified from ``flax.nnx.nn.dtypes.canonicalize_dtype`` to support ``IrrepsArray``
    and list of ``jax.Array``.

    If the ``dtype`` is None this function will infer the dtype. If it is not
    None it will be returned unmodified or an exceptions is raised if the dtype
    is invalid.
    from the input arguments using ``jnp.result_type``.

    Args:
        *args: JAX array compatible values or IrrepsArray. None values
            are ignored.
        dtype: Optional dtype override. If specified the arguments are cast to
            the specified dtype instead and dtype inference is disabled.
        inexact: When True, the output dtype must be a subdtype
        of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
        is useful when you want to apply operations that don't work directly on
        integers like taking a mean for example.
    Returns:
        The dtype that *args should be cast to.
    """
    if dtype is None:
        args_filtered = []
        for x in args:
            if isinstance(x, IrrepsArray):
                args_filtered.append(x.array)
                if x._chunks is not None: # pylint: disable=protected-access
                    # pylint: disable=protected-access
                    args_filtered.extend([chunk for chunk in x._chunks if chunk is not None])
            elif isinstance(x, list):
                args_filtered.extend([jnp.asarray(y) for y in x if y is not None])
            elif x is not None:
                args_filtered.append(jnp.asarray(x))
        dtype = jnp.result_type(*args_filtered)
        if inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(jnp.float32, dtype)

    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError(f'Dtype must be inexact: {dtype}')
    return dtype


def promote_dtype(args: T, /, *, dtype=None, inexact=True) -> T:
    """Promotes input arguments to a specified or inferred dtype.

    Modified from ``flax.nnx.nn.dtypes.promote_dtype`` to support ``IrrepsArray``
    and list of ``jax.Array``.

    All args are cast to the same dtype. See ``canonicalize_dtype`` for how
    this dtype is determined.

    The behavior of promote_dtype is mostly a convenience wrapper around
    ``jax.numpy.promote_types``. The differences being that it automatically casts
    all input to the inferred dtypes, allows inference to be overridden by a
    forced dtype, and has an optional check to guarantee the resulting dtype is
    inexact.

    Args:
        *args: JAX array compatible values or IrrepsArray. None values
            are returned as is.
        dtype: Optional dtype override. If specified the arguments are cast to
            the specified dtype instead and dtype inference is disabled.
        inexact: When True, the output dtype must be a subdtype
        of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
        is useful when you want to apply operations that don't work directly on
        integers like taking a mean for example.
    Returns:
        The arguments cast to arrays of the same dtype.
    """
    dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)

    def _promote(x):
        if x is None:
            return None
        if isinstance(x, IrrepsArray):
            return x.astype(dtype)
        if isinstance(x, list):
            return [None if y is None else jnp.asarray(y, dtype=dtype) for y in x]
        return jnp.asarray(x, dtype=dtype)

    arrays = tuple(_promote(x) for x in args)
    return arrays  # type: ignore[return-value]
