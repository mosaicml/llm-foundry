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
trt_gpt_config = {
    'run_name': 'trtllm-eval',
    'seed': 0,
    'max_seq_len': 1024,
    'device_eval_batch_size': 4,
    'precision': 'amp_fp16',
    'dist_timeout': 6000,
    'models': 
    [ 
        { 
            'model_name': 'trtllm/gpt',
            'model':
            {
                'name': 'trtllm',
                'version': 'gpt',
                'engine_dir': f'{os.environ['TENSORRT_LLM_PATH']}/examples/gpt/gpt2-xl',
                'log_level': 'error'
            },
            'tokenizer': 
            {
                'name': 'gpt2'
            }
        } 
    ],
    'icl_tasks': './eval/yamls/tasks_mini.yaml',
    'model_gauntlet': './eval/yamls/model_gauntlet_mini.yaml',
    'loggers': {
        'wandb': {
            'project': 'julian-trt-benchmarking-gpt2-xl'
        }
    }
}

# trt_llama_config = {
#     'run_name': 'trtllm-eval',
#     'seed': 0,
#     'max_seq_len': 2048,
#     'device_eval_batch_size': 4,
#     'precision': 'amp_bf16',
#     'dist_timeout': 6000,
#     'models': 
#     [ 
#         { 
#             'model_name': 'trtllm/llama',
#             'model':
#             {
#                 'name': 'trtllm',
#                 'version': 'llama',
#                 'engine_dir': {YOUR_LLAMA_ENGINE_DIR},
#                 'log_level': 'error',
#                 'eos_token_id': 2,
#                 'pad_token_id': 2
#             },
#             'tokenizer': 
#             {
#                 'name': '{PATH_TO_HF_MODEL_DIR_OR_HF_HUB_NAME}'
#             }
#         } 
#     ],
#     'icl_tasks': '{PATH_TO_LLMFOUNDRY}/llm-foundry/scripts/eval/yamls/tasks.yaml',
#     'model_gauntlet': '{PATH_TO_LLMFOUNDRY}/llm-foundry/scripts/eval/yamls/model_gauntlet.yaml',
#     'loggers': {
#         'wandb': {
#             'project': '{YOUR_WANDB_PROJECT_NAME}'
#         }
#     }
# }

om_dict_config: DictConfig = om.create(trt_gpt_config)
print("OmegaConfig dictionary", om.to_yaml(om_dict_config))

run_evaluation(om_dict_config)