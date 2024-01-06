# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from composer.utils import (get_file, maybe_create_object_store_from_uri,
                            parse_uri, safe_torch_load)

from llmfoundry.models.mpt.configuration_mpt import (attn_config_defaults,
                                                     init_config_defaults)

# define state dict key changes
# old_state_dict_key: new_state_dict_key
v004_to_llmfoundry_key_conversion = OrderedDict([
    ('.ln_', '.norm_'),
    ('.mlp_up.', '.up_proj.'),
    ('.mlp_down.', '.down_proj.'),
    ('.mlp.', '.ffn.'),
])


def convert_examples_ckpt_state_dict(
    state_dict: Dict[str, Any],
    conversion_dict: Dict[str, str],
) -> Dict[str, Any]:
    # map old keys to new keys
    key_mappings = OrderedDict()
    for k in state_dict.keys():
        key_mappings[k] = k
    for k, v in key_mappings.items():
        _v = v
        for old, new in conversion_dict.items():
            _v = _v.replace(old, new)
        key_mappings[k] = _v

    # generate state dict with new keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if key_mappings[k] != k:
            print(f'Updating state dict key: {k} -> {key_mappings[k]}')
        new_state_dict[key_mappings[k]] = v

    return new_state_dict


def convert_examples_ckpt(
    checkpoint_path: Union[Path, str],
    output_path: Union[Path, str],
    conversion_dict: Dict[str, str],
    local_ckpt_path: Optional[Union[Path, str]] = None,
) -> None:
    """Convert a ckpt created in examples repo to an llmfoundry compat ckpt.

    Args:
        checkpoint_path (Union[Path, str]): Path to the composer checkpoint, can
            be a local path, or a remote path beginning with ``s3://``, or
            another backend supported by
            :meth:`composer.utils.maybe_create_object_store_from_uri`.
        output_path (Union[Path, str]): Path to the folder to write the output to.
            Can be a local path, or a remote path beginning with ``s3://``, or
            another backend supported by
            :meth:`composer.utils.maybe_create_object_store_from_uri`.
        conversion_dict (Dict): defines state dict key changes
        local_ckpt_path (Optional[Union[Path, str]], optional): If specified,
            where to save the checkpoint file to locally. If the input
            ``checkpoint_path`` is already a local path, this will be a
            symlink. Defaults to None, which will use a temporary file.
    """
    # default local path to a tempfile if path is not provided
    if local_ckpt_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        local_ckpt_path = Path(tmp_dir.name) / 'local-composer-checkpoint.pt'

    # create object store if output_path
    _, _, local_folder_path = parse_uri(str(output_path))
    object_store = maybe_create_object_store_from_uri(str(output_path))
    if object_store is not None:
        local_output_path = tempfile.TemporaryDirectory().name
    else:
        local_output_path = local_folder_path

    # create folder
    os.makedirs(local_output_path)

    # download the checkpoint file
    print(f'Downloading checkpoint from {checkpoint_path} -> {local_ckpt_path}')
    get_file(str(checkpoint_path), str(local_ckpt_path))

    # Load the Composer checkpoint state dict
    print('Loading checkpoint into CPU RAM...')
    composer_state_dict = safe_torch_load(local_ckpt_path)

    # Convert examples model state dict to llm-foundry
    model_state = convert_examples_ckpt_state_dict(
        composer_state_dict['state']['model'],
        conversion_dict,
    )
    composer_state_dict['state']['model'] = model_state

    # Convert HF config in state dict
    if 'huggingface' in composer_state_dict['state']['integrations']:
        hf_config = composer_state_dict['state']['integrations']['huggingface'][
            'model']['config']['content']

        if hf_config['model_type'] == 'mosaic_gpt':
            hf_config['model_type'] = 'mpt'

        if 'mlp_ratio' in hf_config:
            hf_config['expansion_ratio'] = hf_config.pop('mlp_ratio')

        # Convert attention config
        if 'attn_config' not in hf_config:
            hf_config['attn_config'] = deepcopy(attn_config_defaults)
            hf_config['attn_config']['attn_type'] = 'multihead_attention'
            hf_config['attn_config']['qk_ln'] = hf_config.pop(
                'attn_qk_ln', attn_config_defaults['qk_ln'])
            hf_config['attn_config']['clip_qkv'] = hf_config.pop(
                'attn_clip_qkv', attn_config_defaults['clip_qkv'])

            for k in [
                    'attn_pdrop', 'attn_impl', 'softmax_scale', 'prefix_lm',
                    'attn_uses_sequence_id', 'alibi', 'alibi_bias_max'
            ]:
                if k in hf_config:
                    hf_config['attn_config'][k] = hf_config.pop(k)

        # convert norm config
        if 'low_precision_layernorm' in hf_config:
            if hf_config.pop('low_precision_layernorm'):
                hf_config['norm_type'] = 'low_precision_layernorm'
            else:
                hf_config['norm_type'] = 'layernorm'

        # Convert init config
        if 'init_config' not in hf_config:
            hf_config['init_config'] = deepcopy(init_config_defaults)
            hf_config['init_config']['name'] = hf_config.pop('param_init_fn')

            for k in [
                    'fan_mode', 'init_nonlinearity', 'init_gain', 'init_std',
                    'init_div_is_residual', 'emb_init_std',
                    'emb_init_uniform_lim'
            ]:
                if k in hf_config:
                    hf_config['init_config'][k] = hf_config.pop(k)

        print(f'Setting hf_config: {hf_config}')
        composer_state_dict['state']['integrations']['huggingface']['model'][
            'config']['content'] = hf_config

    # Convert optimizer state dict
    if 'optimizers' in composer_state_dict['state'].keys():
        print(f'Updating optimizer state dict')
        for opt in composer_state_dict['state']['optimizers'].keys():

            opt_state = convert_examples_ckpt_state_dict(
                composer_state_dict['state']['optimizers'][opt]['state'],
                conversion_dict,
            )
            composer_state_dict['state']['optimizers'][opt]['state'] = opt_state

            for pg_idx in range(
                    len(composer_state_dict['state']['optimizers'][opt]
                        ['param_groups'])):
                for param_idx in range(
                        len(composer_state_dict['state']['optimizers'][opt]
                            ['param_groups'][pg_idx]['params'])):
                    param_name = composer_state_dict['state']['optimizers'][
                        opt]['param_groups'][pg_idx]['params'][param_idx]
                    for old, new in conversion_dict.items():
                        param_name = param_name.replace(old, new)
                    composer_state_dict['state']['optimizers'][opt][
                        'param_groups'][pg_idx]['params'][
                            param_idx] = param_name

    # Save weights
    file_path = str(
        Path(local_output_path) / str(checkpoint_path).split('/')[-1])
    print(f'Writing converted output to {file_path}')
    torch.save(composer_state_dict, file_path)

    if object_store is not None:
        remote_file_path = os.path.join(local_folder_path,
                                        str(checkpoint_path).split('/')[-1])
        print(f'Uploading from {file_path} to {remote_file_path}')
        object_store.upload_object(remote_file_path, file_path)


def main(args: Namespace) -> None:
    convert_examples_ckpt(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        conversion_dict=v004_to_llmfoundry_key_conversion,
        local_ckpt_path=args.local_ckpt_path,
    )


if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        'Convert ckpt created with the examples repo into one usable by llmfoundry.'
    )
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--local_ckpt_path', type=str, default=None)

    args = parser.parse_args()

    main(args)
