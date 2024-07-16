# This file lives in llmfoundry/scripts/eval/evaluate_trtllm_test.py
# It is not included in the PR because it should not be committed.
# It must be called from llmfoundry/scripts directory, like so: composer -n 1 eval/evaluate_trtllm_test.py
# Multi-GPU inference does not work, you must run with -n 1 or CUDA_VISIBLE_DEVICES={A single gpu number}
# All this can be written in YAML form.


from eval import main as run_evaluation
#from eval_trt_multigpu import main as run_evaluation
from omegaconf import OmegaConf as om
from omegaconf import DictConfig


trt_folder_path = '/workspace/TensorRT-LLM/'

MINI_TASKS = './eval/yamls/mini_tasks_v0.2.yaml'
QA_MC_TASKS = './eval/yamls/qa_mc_tasks_v0.3.yaml'
ALL_TASKS = './eval/yamls/tasks_v0.3.yaml'
LM_TASKS = './eval/yamls/lm_tasks_v0.3.yaml'
BROKEN_TASKS = './eval/yamls/broken_tasks.yaml'
EVAL_GAUNTLET = './eval/yamls/eval_gauntlet_v0.3.yaml'

def get_dbrx_config(engine_dir, tokenizer_name, icl_tasks=MINI_TASKS):
    return {
        'run_name': 'trtllm-eval',
        'seed': 0,
        'max_seq_len': 2048,
        'device_eval_batch_size': 4,
        'precision': 'amp_bf16',
        'dist_timeout': 6000,
        'models':
        [
            {
                'model_name': 'trtllm/dbrx',
                'model':
                {
                    'name': 'trtllm',
                    'version': 'dbrx',
                    'engine_dir': engine_dir,
                    'log_level': 'error',
                },
                'tokenizer':
                {
                    'name': tokenizer_name,
                    'kwargs':
                    {
                        'trust_remote_code': 'True'
                    }
                }
            }
        ],
        'icl_tasks': icl_tasks,
        'eval_gauntlet': EVAL_GAUNTLET,
        'loggers': {
            'wandb': {
                'project': 'nik-dbrx-quant-eval'
            }
        }
    }



def get_llama_config(engine_dir, tokenizer_name, icl_tasks=MINI_TASKS):
    return {
        'run_name': 'trtllm-eval',
        'seed': 0,
        'max_seq_len': 1024,
        'device_eval_batch_size': 4, # Llama-7B should be batch size 32
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
                    'engine_dir': engine_dir,
                    'log_level': 'error',
                    'end_token_id': 128009,
                    'pad_token_id': 128001,
 
                },
                'tokenizer':
                {
                    'name': tokenizer_name,
               }
            }
        ],
        'icl_tasks': icl_tasks,
        'eval_gauntlet': EVAL_GAUNTLET,
        'loggers': {
            'wandb': {
                'project': 'nik-llama3-eval'
            }
        }
    }

def engine_dir_str(model_type, model_dir, variant, ngpus=8):
    return f"{trt_folder_path}examples/{model_type}/tmp/{model_type}/{model_dir}/trt_engines/{variant}/{ngpus}-gpu"

LLAMA_TOK_DIR = '/mnt/workdisk/nikhil/models/llama3-70b-instruct-hf/'
DBRX_TOK_DIR = '/mnt/workdisk/nikhil/models/dbrx-hf/03_25_hf_ckpt/'

# LLama URLs
#llama_bf16_engine_dir = '/mnt/workdisk/nikhil/engines/llama3_70b_bf16_logits_v0.10'
#llama_fp8_engine_dir = '/mnt/workdisk/nikhil/engines/llama3_70b_fp8_logits_0521/'
llama_fp8_engine_dir = '/mnt/workdisk/nikhil/engines/llama3_70b_fp8_logits_v2_v0.10'
llama70b_config = get_llama_config(llama_fp8_engine_dir, LLAMA_TOK_DIR)

#dbrx_bf16_engine_dir = '/mnt/workdisk/nikhil/engines/dbrx_bf16_logits_0521/'
#dbrx_bf16_config = get_dbrx_config(dbrx_bf16_engine_dir, DBRX_TOK_DIR)

dbrx_fp8_engine_dir = '/mnt/workdisk/nikhil/engines/dbrx_fp8_logits_v2_v0.10' 
dbrx_fp8_config = get_dbrx_config(dbrx_fp8_engine_dir, DBRX_TOK_DIR)

def run_eval(config):
    print("RUNNING EVAL")
    om_dict_config: DictConfig = om.create(config)
    print("OmegaConfig dictionary", om.to_yaml(om_dict_config))
    run_evaluation(om_dict_config)

# run_eval(dbrx_fp8_config)
run_eval(llama70b_config)

