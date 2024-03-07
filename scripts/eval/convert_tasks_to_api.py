from copy import deepcopy
import json
import sys
import os
from omegaconf import OmegaConf as om
from pathlib import Path

API_DIR = 'eval/local_data/api_compatible_gauntlet'

def process_row(sample, task_type):
    if task_type == 'question_answering' or task_type =='language_modeling' or task_type == 'code_evaluation':
        # we already natively support these
        return sample
    else:
        breakpoint()

def process_task_cfg(task, label, task_type, uri):
    path = Path(uri)
    parent = path.parent.absolute().name
    new_uri = f"{API_DIR}/{parent}/{os.path.basename(uri)}"
    if task_type == 'question_answering' or task_type == 'language_modeling' or task_type == 'code_evaluation':
        task['dataset_uri'] = new_uri
    elif task_type == 'multiple_choice':
        breakpoint()
    elif task_type == 'schema':
        breakpoint()


def process_data(uri, task_type):
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
                    processed_sample = process_row(sample, task_type)
                    out.write((json.dumps(processed_sample, ensure_ascii=False)+ '\n').encode('utf8') )
    else:
        raise NotImplementedError('Data must be in eval/local_data')

if __name__ == "__main__":
    assert len(sys.argv) > 1
    old_tasks_path = sys.argv[1]
    with open(old_tasks_path) as f:
        old_tasks_cfg = om.load(f)

    new_tasks_cfg = deepcopy(old_tasks_cfg)
    for task in old_tasks_cfg['icl_tasks']:
        label = task['label']
        uri = task['dataset_uri']
        task_type = task['icl_task_type']

        process_data(uri, task_type)
        process_task_cfg(task, label, task_type, uri)

        
       
        
