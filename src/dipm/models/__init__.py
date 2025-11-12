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

from dipm.models.predictor import ForceFieldPredictor
from dipm.models.mace.models import Mace
from dipm.models.nequip.models import Nequip
from dipm.models.visnet.models import Visnet
from dipm.models.liten.models import LiTEN
from dipm.models.equiformer_v2.models import EquiformerV2
from dipm.models.uma.models import UMA

KNOWN_MODELS = {
    "Mace": Mace,
    "Nequip": Nequip,
    "Visnet": Visnet,
    "LiTEN": LiTEN,
    "EquiformerV2": EquiformerV2,
    "UMA": UMA,
}
