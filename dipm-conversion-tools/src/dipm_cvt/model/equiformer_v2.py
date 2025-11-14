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

from typing import TYPE_CHECKING
import json
import logging
import re

from dipm_cvt._lazy_loader import LazyLoader
from dipm_cvt.model.utils import StateDictConverter

torch = LazyLoader(
    "torch", globals(), "torch",
    error_msg="`torch` is not installed but required for EquiformerV2 model conversion."
)

safetensors_torch = LazyLoader(
    "safetensors", globals(), "safetensors.torch",
    error_msg="`safetensors` is not installed but required for EquiformerV2 model conversion."
)

if TYPE_CHECKING:
    import torch
    import safetensors.torch as safetensors_torch

logger = logging.getLogger('dipm-cvt-cli')


def convert_hyperparams(config: dict, dataset_config: dict) -> dict:
    """Convert the hyperparameters."""

    # In equiformer_v2, neutron (atomic number 0) is also included.
    num_species = config.pop('num_species', 90) - 1
    dataset_info = {
        'atomic_energies_map': {i: 0.0 for i in range(1, num_species+1)},
        'cutoff_distance_angstrom': config.pop('max_radius', 5.0),
        'avg_num_neighbors': 23.395238876342773, # OC20
        'avg_num_nodes': 77.81317, # OC20
        'avg_r_min_angstrom': None,
        'scaling_mean': dataset_config.get('target_mean', 0.0),
        'scaling_stdev': dataset_config.get('target_std', 1.0),
    }

    # Rename keys
    config['force_head'] = config.pop('regress_forces', True)
    config['num_species'] = num_species
    if config.get('edge_channels', None) is not None:
        config['num_edge_channels'] = config.pop('edge_channels')
    if config.pop('num_distance_basis', None) is not None:
        # In equiformer_v2, whatever the `num_distance_basis`, it is always 600.
        config['num_rbf'] = 600
    use_atom_edge_embedding = config.pop('use_atom_edge_embedding', True)
    share_atom_edge_embedding = config.pop('share_atom_edge_embedding', False)
    if use_atom_edge_embedding:
        if share_atom_edge_embedding:
            config['atom_edge_embedding'] = 'shared'
        else:
            config['atom_edge_embedding'] = 'isolated'
    else:
        config['atom_edge_embedding'] = 'none'

    # Support partially
    if config.pop('distance_function', 'gaussian') != 'gaussian':
        logger.critical("Distance function other than gaussian is not supported.")
    if config.pop('proj_drop', 0.0) != 0.0:
        logger.critical("Projected dropout is not supported.")
    lmax_list = config.pop('lmax_list', [6])
    if len(lmax_list) != 1:
        logger.critical("Multiple lmax resolutions is supported.")
    config['lmax'] = lmax_list[0]
    mmax_list = config.pop('mmax_list', [2])
    if len(mmax_list) != 1:
        logger.critical("Multiple mmax resolutions is supported.")
    config['mmax'] = mmax_list[0]

    # Not used
    config.pop('attn_activation', None)
    config.pop('ffn_activation', None)
    config.pop('use_pbc', None)
    config.pop('otf_graph', None)
    config.pop('use_s2_act_attn', None)
    config.pop('weight_init', None)
    # In fact, `max_neighbors` is always 200, see `AtomsToGraphs`
    config.pop('max_neighbors', None)
    config.pop('num_sphere_samples', None)

    hyperparams = {
        'config': json.dumps(config),
        'dataset_info': json.dumps(dataset_info),
        'predict_stress': 'false',
        'target': 'EquiformerV2',
    }

    return hyperparams


def convert_params(params: dict) -> dict:
    """Convert the model parameters."""

    def transpose(x):
        return x[0].transpose(-1, -2).contiguous()

    def exclude_neutron(x):
        return x[0][1:]

    StateDictConverter({
        r'\.mappingReduced\.': None,
        r'\.mapping\.': None,
        r'\.to_grid_mat$': None,
        r'\.from_grid_mat$': None,
        r'\.expand_index$': None,
        r'\.distance_expansion\.offset$': None,
        r'\.norm([^\.]*)\.balance_degree_weight$': None,
        r'^module\.module\.blocks\.': r'equiformer_model.layers.',
        r'\.source_embedding\.weight$': (r'.senders_embedding.embedding', exclude_neutron),
        r'\.target_embedding\.weight$': (r'.receivers_embedding.embedding', exclude_neutron),
        r'\.sphere_embedding\.weight$': (r'.sphere_embedding.embedding', exclude_neutron),
        r'\.(\d+)\.ga\.': r'.\1.graph_attn.',
        r'^module\.module\.': r'equiformer_model.',
        r'mlp\.(\d+)\.weight$': (r'mlp.layers.\1.kernel', transpose),
        r'mlp\.(\d+)\.bias$': r'mlp.layers.\1.bias',
        r'linear([^\.]*)\.weight$': (r'linear\1.kernel', transpose),
        r'fc([^\.]*)\.weight$': (r'fc\1.kernel', transpose),
        r'\.proj\.weight$': (r'.proj.kernel', transpose),
        r'\.alpha_norm\.weight$': r'.alpha_norm.scale',
        r'\.norm_l0\.weight$': r'.norm_l0.scale',
    }).apply(params)

    # Make MLP layers matched
    for key in list(params.keys()):
        match = re.search(r'\.rad_func\.net\.(\d+)', key)
        if match:
            i = int(match.group(1))
            # Linear layer
            if i % 3 == 0:
                new_key = key.replace(f'.rad_func.net.{i}', f'.rad_func.layers.{i // 3}')
                if new_key.endswith('.weight'):
                    new_key = new_key.replace(
                        f'.layers.{i // 3}.weight', f'.layers.{i // 3}.kernel'
                    )
                    params[new_key] = params.pop(key).transpose(-1, -2).contiguous()
                else:
                    params[new_key] = params.pop(key)
            # Normalization layer
            elif i % 3 == 1:
                new_key = key.replace(f'.rad_func.net.{i}', f'.rad_func.norms.{i // 3}')
                new_key = new_key.replace(f'.norms.{i // 3}.weight', f'.norms.{i // 3}.scale')
                params[new_key] = params.pop(key)
            # Activation layer, impossible to reach here

    return params


def convert_equiformer_v2(from_path: str, to_path: str):
    """Convert an EquiformerV2 model from PyTorch to Safetensors."""
    state_dict = torch.load(from_path, map_location="cpu", weights_only=True)
    if state_dict['config']['model'] != 'equiformer_v2':
        logger.critical("The checkpoint is not from an EquiformerV2 model.")

    config = state_dict['config']['model_attributes']
    dataset_config = state_dict['config']['dataset']
    params = state_dict['state_dict']

    params = convert_params(params)
    hyperparams = convert_hyperparams(config, dataset_config)

    safetensors_torch.save_file(params, to_path, metadata=hyperparams)
