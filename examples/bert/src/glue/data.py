# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import logging

from composer.utils import MissingConditionalImportError, dist

_task_column_names = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
}

log = logging.getLogger(__name__)


def create_glue_dataset(
    task: str,
    tokenizer_name: str,
    split: str,
    max_seq_length: int = 256,
    max_retries: int = 10,
    num_workers: int = 0,
):
    try:
        import datasets
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if task not in _task_column_names:
        raise ValueError(
            f'task ({task}) must be one of {_task_column_names.keys()}')

    if (max_seq_length % 8) != 0:
        log.warning(
            'For performance, a max_seq_length as a multiple of 8 is recommended.'
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name)  #type: ignore (thirdparty)

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        'glue',
        task,
        split=split,
        download_config=download_config,
    )

    log.info(
        f'Starting tokenization by preprocessing over {num_workers} threads!')
    text_column_names = _task_column_names[task]

    def tokenize_function(inp):
        # truncates sentences to max_length or pads them to max_length

        first_half = inp[text_column_names[0]]
        second_half = inp[
            text_column_names[1]] if text_column_names[1] in inp else None
        return tokenizer(
            text=first_half,
            text_pair=second_half,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
        )

    columns_to_remove = ['idx'
                        ] + [i for i in text_column_names if i is not None]

    assert isinstance(dataset, datasets.Dataset)
    safe_name = tokenizer_name.replace('/', ',')
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        new_fingerprint=f'{task}-{safe_name}-tokenization-{split}',
        load_from_cache_file=True,
    )
    return dataset
