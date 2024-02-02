from mcli.sdk import RunConfig, create_run
import copy

base_name = 'eval-callibration'
clusters = ['r7z2', 'r4z8', 'r4z8']
priority = "low"
preemptible = True
retry_on_system_failure = False

token_ratio_to_load_path = {
    7: 'mistralai/Mistral-7B-Instruct-v0.1',
   8: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    70: 'meta-llama/Llama-2-70b-chat-hf',
}

n_gpus = [8, 8, 8] #, 8, 8, 8, 8]
token_ratios = [7, 8, 70] #20, 50, 100, 250, 500]

for c, n_gpu, token_ratio in zip(clusters, reversed(n_gpus), reversed(token_ratios)):
    config = RunConfig.from_file('base_callibration.yaml')
    parameters = copy.deepcopy(config.parameters)
    config.name = f'{base_name}-{token_ratio}'
    config.gpu_num = n_gpu
    config.cluster = c
    config.scheduling = {'priority': priority, 'preemptible': preemptible}
    if retry_on_system_failure:
        config.scheduling = config.scheduling | {'retry_on_system_failure': False, 'max_retries': 1}

    config.integrations.append({
        'integration_type' : 'wandb',
        'project' : 'eval-llama2-callibrate',
        'group': f'{token_ratio}',
        'entity': 'mosaic-ml'})

    run_params = copy.deepcopy(parameters)

    run_params['models'][0]['model_name'] = token_ratio_to_load_path[token_ratio]
    run_params['models'][0]['model']['pretrained_model_name_or_path'] = token_ratio_to_load_path[token_ratio]
    run_params['models'][0]['tokenizer']['name'] = token_ratio_to_load_path[token_ratio]

    config.parameters = run_params
    print(config.name)
    print(config.gpu_num)

    run = create_run(config)
    print(run)