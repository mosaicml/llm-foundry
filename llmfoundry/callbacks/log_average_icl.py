# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""After each eval, log the average performance across all ICL tasks"""
from typing import Union
import copy
from collections import defaultdict
import re
from composer.core import Callback, State, Event
from composer.loggers import Logger
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class AverageICLLogger(Callback):

    def run_event(self, event: Event, state: State, logger: Logger):
        if event != Event.EVAL_STANDALONE_END or event != Event.EVAL_AFTER_ALL:
            return

        eval_metrics = copy.deepcopy(state.eval_metrics)
        num_shot_avgs = defaultdict(list)
        for metric_name, metrics in eval_metrics.items():
            for _, metric_val in metrics.items():
                match = re.search(r"(\d+)-shot", metric_name)
                if not match:
                    continue
                num_shots = int(match.group(1))
                num_shot_avgs[num_shots].append(metric_val.compute())
        if len(num_shot_avgs) == 0:
            return
        num_shot_avgs = {
            f"metrics/icl/{num_shot}-shot/avg": sum(perfs) / len(perfs)
            for num_shot, perfs in num_shot_avgs.items()
        }
        logger.log_metrics(num_shot_avgs)
