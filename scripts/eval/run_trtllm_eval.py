# This file lives in llmfoundry/scripts/eval/evaluate_trtllm_test.py
# It is not included in the PR because it should not be committed.
# It must be called from llmfoundry/scripts directory, like so: composer -n 1 eval/evaluate_trtllm_test.py
# Multi-GPU inference does not work, you must run with -n 1 or CUDA_VISIBLE_DEVICES={A single gpu number}
# All this can be written in YAML form.


from eval_trt_multigpu import main as run_evaluation
from omegaconf import OmegaConf as om
from omegaconf import DictConfig


trt_folder_path = '/workspace/TensorRT-LLM/'

MINI_TASKS = './eval/yamls/mini_tasks_v0.2.yaml'
QA_MC_TASKS = './eval/yamls/qa_mc_tasks_v0.2.yaml'
ALL_TASKS = './eval/yamls/tasks_v0.2.yaml'
LM_TASKS = './eval/yamls/lm_tasks_v0.2.yaml'

# GPT config is just for quick initial testing purposes
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
                'engine_dir': trt_folder_path + 'examples/gpt/engine_outputs',
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


def get_llama_config(engine_dir, tokenizer_name, icl_tasks=MINI_TASKS):
    return {
        'run_name': 'trtllm-eval',
        'seed': 0,
        'max_seq_len': 2048,
        'device_eval_batch_size': 8, # Llama-7B should be batch size 32
        'precision': 'amp_fp16',
        'dist_timeout': 6000,
        'models':
        [
            {
                'model_name': 'trtllm/llama',
                'model':
                {
                    'name': 'trtllm',
                    'version': 'llama',
                    'engine_dir': engine_dir,
                    'log_level': 'error',
                    'eos_token_id': 2,
                    'pad_token_id': 2
                },
                'tokenizer':
                {
                    'name': tokenizer_name,
                }
            }
        ],
        'icl_tasks': icl_tasks,
        'eval_gauntlet': './eval/yamls/eval_gauntlet_v0.2.yaml',
        'loggers': {
            'wandb': {
                'project': 'nik-quant-eval'
            }
        }
    }

def engine_dir_str(model_type, model_dir, variant, ngpus=8):
    return f"{trt_folder_path}examples/{model_type}/tmp/{model_type}/{model_dir}/trt_engines/{variant}/{ngpus}-gpu"


LLAMA_TOK_DIR = '/workspace/llama-70b-chat-hf/'
LLAMA_7B_DIR = '7B-chat-quality-eval'
LLAMA_70B_DIR = '70B-chat-quality-eval'


llama7b_int8_config = get_llama_config(engine_dir_str('llama', LLAMA_7B_DIR, 'int8_kv_cache_weight_only', 1), LLAMA_TOK_DIR)
llama70b_fp8_config = get_llama_config(engine_dir_str('llama', LLAMA_70B_DIR, 'fp8'), LLAMA_TOK_DIR)
llama70b_int8_config = get_llama_config(engine_dir_str('llama', LLAMA_70B_DIR, 'int8_kv_cache_weight_only'), LLAMA_TOK_DIR)
llama70b_smoothquant_config = get_llama_config(engine_dir_str('llama', LLAMA_70B_DIR, 'sq0.8'), LLAMA_TOK_DIR)
llama70b_fp16_config = get_llama_config(engine_dir_str('llama', LLAMA_70B_DIR, 'fp16'), LLAMA_TOK_DIR)


def run_eval(config):
    print("RUNNING EVAL")
    om_dict_config: DictConfig = om.create(config)
    print("OmegaConfig dictionary", om.to_yaml(om_dict_config))
    run_evaluation(om_dict_config)

#run_eval(llama7b_int8_config)
#run_eval(llama70b_int8_config)
#run_eval(llama70b_fp16_config)
#run_eval(llama70b_fp8_config)
run_eval(llama70b_smoothquant_config)


