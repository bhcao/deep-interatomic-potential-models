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

from enum import Enum
from pathlib import Path
from typing import Annotated

import e3nn_jax as e3nn
from pydantic import BeforeValidator, AfterValidator, Field
import jax.numpy as jnp

PositiveFloat = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Proportion = Annotated[float, Field(ge=0.0, le=1.0)]

PathLike = Annotated[Path, BeforeValidator(lambda p: Path(p).absolute())]


def _check_irreps(irreps: str) -> str:
    """Check that a string can be interpreted as `e3nn.Irreps`.

    This is useful to stop at the validation step.
    We can't return `e3nn.Irreps` for now as `model_dump` would break.
    """
    _ = e3nn.Irreps(irreps)
    return irreps


Irreps = Annotated[str, AfterValidator(_check_irreps)]


_JAX_INEXACT_DTYPES = {
    "float32": jnp.float32,
    "f32": jnp.float32,
    "float16": jnp.float16,
    "f16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "bf16": jnp.bfloat16,
    "float64": jnp.float64,
    "f64": jnp.float64,
}

class DtypeEnum(Enum):
    """Enum of JAX inexact dtypes."""

    FLOAT32 = "float32"
    F32 = "f32"
    FLOAT16 = "float16"
    F16 = "f16"
    BFLOAT16 = "bfloat16"
    BF16 = "bf16"
    FLOAT64 = "float64"
    F64 = "f64"

def get_dtype(name: str | DtypeEnum | None):
    """Get a JAX inexact dtype from a string name."""
    if name is None:
        return None
    if isinstance(name, DtypeEnum):
        name = name.value
    return _JAX_INEXACT_DTYPES[name.lower()]
