# This file lives in llmfoundry/scripts/eval/evaluate_trtllm_test.py
# It is not included in the PR because it should not be committed.
# It must be called from llmfoundry/scripts directory, like so: composer -n 1 eval/evaluate_trtllm_test.py
# Multi-GPU inference does not work, you must run with -n 1 or CUDA_VISIBLE_DEVICES={A single gpu number}
# All this can be written in YAML form.


from eval import main as run_evaluation
from omegaconf import OmegaConf as om
from omegaconf import DictConfig

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
                'engine_dir': '/workspace/tensorrt-llm-private/examples/gpt/engine_outputs',
                'log_level': 'error'
            },
            'tokenizer':
            {
                'name': 'gpt2'
            }
        }
    ],
    'icl_tasks': './eval/yamls/mini_tasks_v0.2.yaml',
    'eval_gauntlet': './eval/yamls/mini_eval_gauntlet_v0.2.yaml',
}

trt_llama_config = {
    'run_name': 'trtllm-eval',
    'seed': 0,
    'max_seq_len': 2048,
    'device_eval_batch_size': 4,
    'precision': 'amp_bf16',
    'dist_timeout': 6000,
    'models':
    [
        {
            'model_name': 'trtllm/llama',
            'model':
            {
                'name': 'trtllm',
                'version': 'llama',
                'engine_dir': '/workspace/tensorrt-llm-private/examples/llama/tmp/trt-models/llama-2-7b-chat/bf16/1-gpu',
                'log_level': 'error',
                'eos_token_id': 2,
                'pad_token_id': 2
            },
            'tokenizer':
            {
                'name': '/workspace/llama-70b-chat-hf/'
            }
        }
    ],
    'icl_tasks': './eval/yamls/tasks_v0.2.yaml',
    'eval_gauntlet': './eval/yamls/eval_gauntlet_v0.2.yaml',
    'loggers': {
        'wandb': {
            'project': 'nik-quant-eval'
        }
    }
}

om_dict_config: DictConfig = om.create(trt_gpt_config)
print("OmegaConfig dictionary", om.to_yaml(om_dict_config))

run_evaluation(om_dict_config)
