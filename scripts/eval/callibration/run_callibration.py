import copy
from mcli.sdk import RunConfig, create_run

# Edit the yaml file to change the model and the benchmarks

YAML_FILE = 'base_callibration.yaml'
NAME = 'eval-callibration'

# Edit the clusters, priority, preemptible, retry_on_system_failure

clusters = ['rxzx', 'rxzx', 'rxzx', 'rxzx']
n_gpus = [8, 8, 8, 8]
priority = "low"
preemptible = True
retry_on_system_failure = False

# Edit the independant variable

independant_variable_to_load_path = {
    7: 'meta-llama/Llama-2-7b-hf',
   13: 'meta-llama/Llama-2-13b-hf',
   70: 'meta-llama/Llama-2-70b-hf',
    71: 'meta-llama/Llama-2-70b-chat-hf',
}
independant_variable = list(independant_variable_to_load_path.keys())

for c, n_gpu, independant_variable in zip(clusters, n_gpus, independant_variable):
    config = RunConfig.from_file(YAML_FILE)
    parameters = copy.deepcopy(config.parameters)
    config.name = f'{NAME}-{independant_variable}'
    config.gpu_num = n_gpu
    config.cluster = c
    config.scheduling = {'priority': priority, 'preemptible': preemptible}
    if retry_on_system_failure:
        config.scheduling = config.scheduling | {'retry_on_system_failure': False, 'max_retries': 1}

    # Edit the wandb integrations
    config.integrations.append({
        'integration_type' : 'wandb',
        'project' : 'eval-llama2-callibrate',
        'group': f'{independant_variable}',
        'entity': 'mosaic-ml'})

    run_params = copy.deepcopy(parameters)

    run_params['models'][0]['model_name'] = independant_variable_to_load_path[independant_variable]
    run_params['models'][0]['model']['pretrained_model_name_or_path'] = independant_variable_to_load_path[independant_variable]
    run_params['models'][0]['tokenizer']['name'] = independant_variable_to_load_path[independant_variable]

    config.parameters = run_params
    print(config.name)
    print(config.gpu_num)

    run = create_run(config)
    print(run)