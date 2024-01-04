# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from llmfoundry.utils import prompt_files as utils


def test_load_prompt_strings(tmp_path: Path):
    assert utils.load_prompts(['hello', 'goodbye']) == ['hello', 'goodbye']

    with open(tmp_path / 'prompts.txt', 'w') as f:
        f.write('hello goodbye')

    temp = utils.PROMPTFILE_PREFIX + str(tmp_path / 'prompts.txt')
    assert utils.load_prompts(
        [temp, temp, 'why'],
        ' ') == ['hello', 'goodbye', 'hello', 'goodbye', 'why']
