# This file lives in llmfoundry/scripts/eval/eval_tensorrt.py
# It is not included in the PR because it should not be committed.
# It must be called from llmfoundry/scripts directory, like so: composer -n 1 eval/eval_tensorrt.py
# Multi-GPU inference does not work, you must run with -n 1 or CUDA_VISIBLE_DEVICES={A single gpu number}
# All this can be written in YAML form.


from eval.eval import main as run_evaluation
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
import os

# Note: GPT-2 gets straight 0s, don't bother eval-ing except as a sanity check.
trt_mpt_config = {
    'run_name': 'trtllm-eval',
    'seed': 0,
    'max_seq_len': 2048,
    'device_eval_batch_size': 4,
    'precision': 'amp_fp16',
    'dist_timeout': 6000,
    'models': [],
    'icl_tasks': './eval/yamls/tasks_mini.yaml',
    'model_gauntlet': './eval/yamls/model_gauntlet_mini.yaml',
    'loggers': {
        'wandb': {
            'project': 'mpt-7b-quantization-eval'
        }
    }
}

base = 'mpt-7b'
variants = [
    'fp16',
    'kvquant',
    'int8',
    'int8_kvquant',
    # 'int4',
    # 'int4_kvquant',
    # 'smoothquant',
    # 'smoothquant_kvquant'
]

for variant in variants:
    trt_mpt_config['models'].append({ 
        'model_name': f'trtllm/{base}_{variant}',
        'model':
        {
            'name': 'trtllm',
            'version': 'gpt',
            'engine_dir': f'{os.environ["TENSORRT_LLM_PATH"]}/examples/mpt/{base}_{variant}',
            'log_level': 'error'
        },
        'tokenizer': 
        {
            'name': 'EleutherAI/gpt-neox-20b'
        }
    })

om_dict_config: DictConfig = om.create(trt_mpt_config)
print("OmegaConfig dictionary", om.to_yaml(om_dict_config))

run_evaluation(om_dict_config)