# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Optional


def make_tiny_ft_dataset(
    path: str,
    size: int = 4,
    add_bad_data_dropped: bool = False,
    add_bad_data_error: bool = False,
    add_just_bos_eos_pad: bool = False,
    pad_token: Optional[str] = None,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
):
    good_sample = {'prompt': 'hello', 'response': 'goodbye'}
    samples = [good_sample] * size
    if add_bad_data_dropped:
        if pad_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_bad_data is True'
            )
        # empty prompt
        samples.append({'prompt': '', 'response': 'goodbye'})
        # empty response
        samples.append({'prompt': 'hello', 'response': ''})
        # response just pad
        samples.append({'prompt': 'hello', 'response': pad_token})
        # response just pad multiple times
        samples.append({'prompt': 'hello', 'response': pad_token * 3})

    if add_bad_data_error:
        # prompt just None
        samples.append({
            'prompt': None,
            'response': 'goodbye'
        })  # type: ignore (intentional test)
        # response just None
        samples.append({
            'prompt': 'hello',
            'response': None
        })  # type: ignore (intentional test)

    if add_just_bos_eos_pad:
        if pad_token is None or start_token is None or end_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_just_bos_eos is True'
            )
        # prompt just start
        samples.append({'prompt': start_token, 'response': 'goodbye'})
        # response just start
        samples.append({'prompt': 'hello', 'response': start_token})
        # prompt just end
        samples.append({'prompt': end_token, 'response': 'goodbye'})
        # response just end
        samples.append({'prompt': 'hello', 'response': end_token})
        # prompt just pad
        samples.append({'prompt': pad_token, 'response': 'goodbye'})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as _f:
        for sample in samples:
            _f.write(json.dumps(sample))
            _f.write('\n')
