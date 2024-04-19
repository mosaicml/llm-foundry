# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import sys
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

API_DIR = 'eval/local_data/api_compatible_gauntlet'

# example_delimiter


def process_row(sample: dict, task_type: str, task_cfg: DictConfig):
    if task_type == 'question_answering' or task_type == 'language_modeling' or task_type == 'code_evaluation' or task_type == 'generation_task_with_answers':
        # we already natively support these
        return sample
    elif task_type == 'multiple_choice':
        query = sample['query']
        choices = sample['choices']
        gold_idx = sample['gold']
        cont_delim = task_cfg.get('continuation_delimiter', ' ')
        alphabet = ['(A)', '(B)', '(C)', '(D)']
        alphabet = alphabet[0:len(choices)]
        choices_str = '\n'.join([f'{a} {c}' for a, c in zip(alphabet, choices)])
        if 'Choices' in query:
            query, orig_choices = query.split('Choices')[0], query.split(
                'Choices')[1]
            orig_choices = orig_choices.replace('A.', '(A)').replace(
                'B.', '(B)').replace('C.', '(C)').replace('D.', '(D)')
            choices_str = orig_choices.strip()

        query = re.sub('(Question:|Q:) ?', '', query)
        query = re.sub('(Answer:|A:) ?', '', query).strip()
        context = f'{query}\n{choices_str}' + '\nA:'
        if 'Question:' not in context and 'Q:' not in context:
            context = 'Q: ' + context
        answer = alphabet[gold_idx]

        if 'category' in sample:
            return {
                'context': context.strip(),
                'answer': answer.strip(),
                'category': sample['category']
            }
        else:
            return {'context': context.strip(), 'answer': answer.strip()}
    elif task_type == 'schema':
        context_options = sample['context_options']
        gold_idx = sample['gold']
        continuation = sample['continuation']
        prompt = 'Read the following choices and determine which is better\n'
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        alphabet = alphabet[0:len(context_options)]

        context_options_str = 'Choices:\n' + '\n'.join([
            f'{a}. {c} {continuation}'
            for a, c in zip(alphabet, context_options)
        ])

        if 'Question:' not in context_options_str and 'Q:' not in context_options_str:
            context = 'Question: ' + context_options_str

        prompt += context_options_str
        answer = alphabet[gold_idx]
        prompt = re.sub('(Answer:|A:)', '', prompt)
        if 'category' in sample:
            return {
                'context': prompt.strip(),
                'answer': answer.strip(),
                'category': sample['category']
            }
        else:
            return {'context': prompt.strip(), 'answer': answer.strip()}


def process_task_cfg(task: DictConfig, task_type: str, uri: str):
    path = Path(uri)
    parent = path.parent.absolute().name
    new_uri = f'{API_DIR}/{parent}/{os.path.basename(uri)}'
    if task_type == 'question_answering' or task_type == 'language_modeling' or task_type == 'code_evaluation':
        task['dataset_uri'] = new_uri
    elif task_type == 'multiple_choice':
        task['dataset_uri'] = new_uri
        task['icl_task_type'] = 'question_answering'
        task['continuation_delimiter'] = '\\n\\nAnswer:'
        task['example_delimiter'] = '\\n\\n'
        task['early_stopping_criteria'] = ['\\n\\n', 'Question:', 'Q:']
    elif task_type == 'schema':
        task['dataset_uri'] = new_uri
        task['icl_task_type'] = 'question_answering'
        task['continuation_delimiter'] = '\\n\\nAnswer:'
        task['example_delimiter'] = '\\n\\n'
        task['early_stopping_criteria'] = ['\\n\\n', 'Question:', 'Q:']

    return task


def process_data(uri: str, task_type: str, task_cfg: DictConfig):
    if uri.startswith('hf://'):
        raise NotImplementedError('URI must be a local path')
    if uri.startswith('eval/local_data'):
        path = Path(uri)
        parent = path.parent.absolute().name

        if not os.path.exists(f'{API_DIR}/{parent}'):
            os.mkdir(f'{API_DIR}/{parent}')

        with open(uri, 'r') as f:
            with open(f'{API_DIR}/{parent}/{os.path.basename(f.name)}',
                      'wb') as out:
                for line in f.readlines():
                    sample = json.loads(line)
                    processed_sample = process_row(sample, task_type, task_cfg)
                    out.write(
                        (json.dumps(processed_sample, ensure_ascii=False) +
                         '\n').encode('utf8'))
    else:
        raise NotImplementedError('Data must be in eval/local_data')


if __name__ == '__main__':
    assert len(sys.argv) > 1
    old_tasks_path = sys.argv[1]
    with open(old_tasks_path) as f:
        old_tasks_cfg = om.load(f)

    task_fn = old_tasks_path.split('/')[-1]
    new_tasks_cfg = {'icl_tasks': []}
    for task in old_tasks_cfg['icl_tasks']:  # pyright: ignore
        label = task['label']
        uri = task['dataset_uri']
        task_type = task['icl_task_type']

        process_data(uri, task_type, task)
        new_tasks_cfg['icl_tasks'].append(process_task_cfg(
            task, task_type, uri))
    with open(f'api_compatible_{task_fn}', 'w') as fp:
        om.save(config=new_tasks_cfg, f=fp.name)
