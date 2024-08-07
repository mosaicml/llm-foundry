# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import tempfile
from datetime import datetime
from typing import List

import torch
import transformers

from llmfoundry import MPTConfig, MPTForCausalLM
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility

_ALL_MODELS = [
    'mosaicml/mpt-7b',
    'mosaicml/mpt-7b-instruct',
    'mosaicml/mpt-7b-chat',
    'mosaicml/mpt-30b',
    'mosaicml/mpt-30b-chat',
    'mosaicml/mpt-30b-instruct',
    'mosaicml/mpt-7b-8k',
    'mosaicml/mpt-7b-8k-instruct',
    'mosaicml/mpt-7b-8k-chat',
    'mosaicml/mpt-7b-storywriter',
]


def main(hf_repos_for_upload: List[str]):
    if len(hf_repos_for_upload) == 1 and hf_repos_for_upload[0] == 'all':
        hf_repos_for_upload = _ALL_MODELS

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%B %d, %Y %H:%M:%S')

    from huggingface_hub import HfApi
    api = HfApi()

    # register config auto class
    MPTConfig.register_for_auto_class()

    # register model auto class
    MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

    config = MPTConfig()
    config.attn_config['attn_impl'] = 'torch'
    loaded_hf_model = MPTForCausalLM(config)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        original_save_dir = os.path.join(_tmp_dir, 'model_current')
        loaded_hf_model.save_pretrained(original_save_dir)

        edit_files_for_hf_compatibility(original_save_dir)

        for repo in hf_repos_for_upload:
            print(f'Testing code changes for {repo}')
            pr_model = transformers.AutoModelForCausalLM.from_pretrained(
                original_save_dir,
                trust_remote_code=True,
                device_map='auto',
            )
            pr_tokenizer = transformers.AutoTokenizer.from_pretrained(
                repo,
                trust_remote_code=True,
            )

            generation = pr_model.generate(
                pr_tokenizer(
                    'MosaicML is',
                    return_tensors='pt',
                ).input_ids.to('cuda' if torch.cuda.is_available() else 'cpu'),
                max_new_tokens=2,
            )
            _ = pr_tokenizer.batch_decode(generation)

            print(f'Opening PR against {repo}')
            result = api.upload_folder(
                folder_path=original_save_dir,
                repo_id=repo,
                use_auth_token=True,
                repo_type='model',
                allow_patterns=['*.py'],
                commit_message=f'LLM-foundry update {formatted_datetime}',
                create_pr=True,
            )

            print(f'PR opened: {result}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Update MPT code in HuggingFace Hub repos to be in sync with the local codebase',
    )
    parser.add_argument(
        '--hf_repos_for_upload',
        help='List of repos to open PRs against',
        nargs='+',
        required=True,
    )

    args = parser.parse_args()

    main(hf_repos_for_upload=args.hf_repos_for_upload,)
