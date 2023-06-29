# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Print Evals."""
from __future__ import annotations

import torch
from composer.core import Callback, State
from composer.loggers import Logger

__all__ = [
    'PrintICLExample',
]


class PrintICLExample(Callback):
    """GlobalLRScaling.

    This callback decodes and prints the first example in a batch,
    for the specified number of consecutive batches.

    Args: num_consecutive_batches_to_print (int):
        prints an example from this many consecutive batches,
        ie. 2 -> prints the first example from 2 consecutive batches
    """

    def __init__(self, num_consecutive_batches_to_print: int = 1):
        self.batches_printed = 0
        self.num_consecutive_batches_to_print = num_consecutive_batches_to_print

    def eval_start(self, state: State, logger: Logger):
        # reset the counter between different eval tasks
        self.batches_printed = 0

    def eval_batch_start(self, state: State, logger: Logger):

        if self.batches_printed < self.num_consecutive_batches_to_print:
            # print example in green
            print('\033[92m' + 'Example: ' + '\033[0m')
            print('-' * 10)

            assert 'input_ids' in state.batch
            # get the first example from the batch
            example = state.batch['input_ids'][0]
            if state.is_model_ddp:
                tokenizer = state.model.module.tokenizer
            else:
                tokenizer = state.model.tokenizer
            print(tokenizer.decode(example, skip_special_tokens=True))
            self.batches_printed += 1
            print('-' * 10)
