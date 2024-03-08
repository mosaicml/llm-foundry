from copy import deepcopy
import json
import sys
import os
from omegaconf import OmegaConf as om 
from pathlib import Path

API_DIR = 'eval/local_data/api_compatible_gauntlet'

def process_row(sample, task_type, task_cfg):
    if task_type == 'question_answering' or task_type =='language_modeling' or task_type == 'code_evaluation':
        # we already natively support these
        return sample
    elif task_type == 'multiple_choice':
        query = sample['query']
        choices = sample['choices']
        gold_idx = sample['gold']
        cont_delim = task_cfg.get('continuation_delimiter', '\n')
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I']
        alphabet = alphabet[0:len(choices)]

        choices_str = "Choices:\n" + '\n'.join([f"{a}. {c}" for a,c in zip(alphabet, choices)])
        context = f"{query}\n{choices_str}{cont_delim}"
        answer = alphabet[gold_idx]
        return {
            "context": context,
            "answer": answer
        }
    
    elif task_type == 'schema':
        context_options = sample['context_options']
        gold_idx = sample['gold']
        continuation = sample['continuation']
        prompt = "Read the following choices and determine which is better\n"
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I']
        alphabet = alphabet[0:len(context_options)]
        context_options_str = "Choices:\n" + '\n'.join([f"{a}. {c} {continuation}" for a,c in zip(alphabet, context_options)])

        prompt += context_options_str
        answer = alphabet[gold_idx]
        return {
            "context": prompt,
            "answer": answer
        }

def process_task_cfg(task, label, task_type, uri):
    path = Path(uri)
    parent = path.parent.absolute().name
    new_uri = f"{API_DIR}/{parent}/{os.path.basename(uri)}"
    if task_type == 'question_answering' or task_type == 'language_modeling' or task_type == 'code_evaluation':
        task['dataset_uri'] = new_uri
    elif task_type == 'multiple_choice':
        task['dataset_uri'] = new_uri
        task['task_type'] = 'question_answering'
        task['continuation_delimiter'] = '\nAnswer:'
    elif task_type == 'schema':
        task['dataset_uri'] = new_uri
        task['task_type'] = 'question_answering'
        task['continuation_delimiter'] = '\nAnswer:'

    
    return task


def process_data(uri, task_type, task_cfg):
    if uri.startswith('hf://'):
        raise NotImplementedError("URI must be a local path")
    if uri.startswith('eval/local_data'):
        path = Path(uri)
        parent = path.parent.absolute().name

        if not os.path.exists(f"{API_DIR}/{parent}"):
            os.mkdir(f"{API_DIR}/{parent}")
        
        with open(uri, "r") as f:
            with open(f"{API_DIR}/{parent}/{os.path.basename(f.name)}", "wb") as out:
                for line in f.readlines():
                    sample = json.loads(line)
                    processed_sample = process_row(sample, task_type, task_cfg)
                    out.write((json.dumps(processed_sample, ensure_ascii=False)+ '\n').encode('utf8') )
    else:
        raise NotImplementedError('Data must be in eval/local_data')

if __name__ == "__main__":
    assert len(sys.argv) > 1
    old_tasks_path = sys.argv[1]
    with open(old_tasks_path) as f:
        old_tasks_cfg = om.load(f)

    task_fn = old_tasks_path.split('/')[-1]
    new_tasks_cfg = {'icl_tasks': []}
    for task in old_tasks_cfg['icl_tasks']:
        label = task['label']
        uri = task['dataset_uri']
        task_type = task['icl_task_type']

        process_data(uri, task_type, task)
        new_tasks_cfg['icl_tasks'].append(
            process_task_cfg(task, label, task_type, uri)
        )
    with open(f'api_comptaible_{task_fn}', 'w') as fp:
        om.save(config=new_tasks_cfg, f=fp.name)

        
       
        
