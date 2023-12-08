# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

from composer.utils import get_file, parse_uri
from datasets import load_dataset

PROMPTFILE_PREFIX = 'file::'
PROMPTDATASET_PREFIX = 'hf://'


def load_prompts(prompts: List[str],
                 prompt_delimiter: Optional[str] = None) -> List[str]:
    """Loads a set of prompts, both free text and from file or HF dataset.

    Args:
        prompts (List[str]): List of free text prompts and prompt files
        prompt_delimiter (Optional str): Delimiter for text file
            If not provided, assumes the prompt file is a single prompt (non-delimited)

    Returns:
        List of prompt string(s)
    """
    prompt_strings = []
    for prompt in prompts:
        backend, _, _ = parse_uri(prompt)
        if prompt.startswith(PROMPTFILE_PREFIX):
            prompts = load_prompts_from_file(prompt, prompt_delimiter)
            prompt_strings.extend(prompts)
        elif prompt.startswith(PROMPTDATASET_PREFIX):
            prompts = load_prompts_from_dataset(prompt, prompt_delimiter)
            prompt_strings.extend(prompts)
        elif backend not in ['', None]:
            prompts = load_prompts_from_remote(prompt, prompt_delimiter)
            prompt_strings.extend(prompts)
        else:
            prompt_strings.append(prompt)
    return prompt_strings


def load_prompts_from_file(prompt_path: str,
                           prompt_delimiter: Optional[str] = None) -> List[str]:
    """Load a set of prompts from a text fie.

    Args:
        prompt_path (str): Path for text file
        prompt_delimiter (Optional str): Delimiter for text file
            If not provided, assumes the prompt file is a single prompt (non-delimited)

    Returns:
        List of prompt string(s)
    """
    if not prompt_path.startswith(PROMPTFILE_PREFIX):
        raise ValueError(f'prompt_path_str must start with {PROMPTFILE_PREFIX}')

    _, prompt_file_path = prompt_path.split(PROMPTFILE_PREFIX, maxsplit=1)
    # local file
    prompt_file_path = os.path.expanduser(prompt_file_path)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(
            f'{prompt_file_path=} does not match any existing files.')

    with open(prompt_file_path, 'r') as f:
        prompt_string = f.read()

    if prompt_delimiter is None:
        return [prompt_string]
    return [i for i in prompt_string.split(prompt_delimiter) if i]


def load_prompts_from_remote(prompt_path: str,
                             prompt_delimiter: Optional[str] = None) -> List[str]:
        """Load a set of prompts from object storage.
    
        Args:
            prompt_path (str): Path for text file
            prompt_delimiter (Optional str): Delimiter for text file
                If not provided, assumes the prompt file is a single prompt (non-delimited)

        Returns:
            List of prompt string(s)
        """
        backend, _, _ = parse_uri(prompt_path)
        if backend in ['', None]:
            raise ValueError(
                f'prompt_path_str must start with s3:// etc if using object storage')

        local_path = prompt_path.split('/')[-1]
        get_file(path=prompt_path, destination=local_path, overwrite=True)

        with open(local_path, 'r') as f:
            prompt_string = f.read()

        if prompt_delimiter is None:
            return [prompt_string]
        return [i for i in prompt_string.split(prompt_delimiter) if i]


def load_prompts_from_dataset(dataset_path: str,
                              prompt_delimiter: Optional[str] = None
                             ) -> List[str]:
    """Load a set of prompts from a huggingface dataset.

    Args:
        dataset_path (str): Path for dataset
        prompt_delimiter (Optional str): We misuse the delimiter here to specify
            the name of the prompt column in the dataset. If not provided, assumes the
            prompt column is named 'prompt'.

    Returns:
        List of prompt string(s)
    """
    if not dataset_path.startswith(PROMPTDATASET_PREFIX):
        raise ValueError(f'dataset_path must start with {PROMPTDATASET_PREFIX}')

    _, dataset_path = dataset_path.split(PROMPTDATASET_PREFIX, maxsplit=1)

    try:
        dataset = load_dataset(dataset_path, token=True)
    except:
        dataset = load_dataset(dataset_path)

    prompt_strings = []
    if prompt_delimiter is None:
        prompt_delimiter = 'prompt'
    try:
        ds = dataset['train']
    except:
        ds = dataset

    if prompt_delimiter not in ds.column_names:
        raise ValueError(f'{prompt_delimiter} not in dataset columns.')
    for prompt in ds[prompt_delimiter]:
        prompt_strings.append(prompt)

    return prompt_strings
