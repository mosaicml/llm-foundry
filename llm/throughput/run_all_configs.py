# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse

import requests
import yaml
from mcli.sdk import RunConfig, create_run

CLUSTER_INFO = {
    # Cluster: [(gpu_type, max_gpus_per_run)],
    'CLUSTER_NAME': [('a100_80gb', 64),],
    'CLUSTER_NAME': [('a100_40gb', 128),],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Generate and run configurations to test MosaicGPT training throughput on Mosaic Cloud.'
    )

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument(
        '--image',
        type=str,
        default='mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04')
    parser.add_argument('-t',
                        '--precisions',
                        '--types',
                        type=str,
                        default=['bf16'],
                        nargs='+',
                        choices=['bf16', 'fp16'])
    parser.add_argument('--fsdp_config_mixed_precision',
                        type=str,
                        default='DEFAULT')
    parser.add_argument(
        '-s',
        '--seq_len_exp',
        type=int,
        default=[9, 14],
        nargs=2,
        help=
        'exponent of seq lengths to be tested (default: [9, 14] = 2^9 to 2^13)')
    parser.add_argument(
        '-b',
        '--batch_size_exp',
        type=int,
        default=[23, 23],
        nargs=2,
        help=
        'exponent of batch size (in tokens) to be tested (default: [19, 23] = 2^19 to 2^23)'
    )
    parser.add_argument(
        '--yaml_base',
        type=str,
        default=
        'https://raw.githubusercontent.com/mosaicml/benchmarks/main/llm/yamls/mosaic_gpt/'
    )
    parser.add_argument('-m',
                        '--model_yamls',
                        type=str,
                        default=[
                            '125m.yaml', '350m.yaml', '760m.yaml', '1b.yaml',
                            '3b.yaml', '7b.yaml', '13b.yaml', '30b.yaml',
                            '70b.yaml'
                        ],
                        choices=[
                            '125m.yaml', '350m.yaml', '760m.yaml', '1b.yaml',
                            '3b.yaml', '7b.yaml', '13b.yaml', '30b.yaml',
                            '70b.yaml'
                        ],
                        nargs='+',
                        help='model sizes to test')

    # NOTE: based on mosaic internal use clusters
    parser.add_argument('-c',
                        '--clusters',
                        type=str,
                        default=['r7z2'],
                        nargs='+',
                        choices=CLUSTER_INFO.keys())
    known_args = parser.parse_known_args()[0]
    _gpu_types = get_gpu_types(known_args.clusters)
    parser.add_argument('--gpu_types',
                        type=str,
                        default=['a100_40gb'],
                        nargs='+',
                        choices=_gpu_types)
    known_args = parser.parse_known_args()[0]
    _gpu_nums = get_gpu_nums(known_args.clusters, known_args.gpu_types)
    parser.add_argument('-g',
                        '--gpu_nums',
                        type=int,
                        default=[16],
                        nargs='+',
                        choices=_gpu_nums)

    parser.add_argument('--microbatch_size',
                        type=int,
                        default=None,
                        help='set microbatch_size')

    parser.add_argument('--disable_wandb', action='store_true')

    parser.add_argument('--RUN', action='store_true')

    return parser.parse_args()


def get_max_seq_lens(pows=[9, 14]):
    return [2**n for n in range(pows[0], pows[1] + 1)]


def get_global_train_batch_sizes(max_seq_len, pows=[19, 23]):
    # global batch size in tokens (defualt: .5M thru 8M)
    global_train_token_counts = [2**n for n in range(pows[0], pows[1] + 1)]
    return [t // max_seq_len for t in global_train_token_counts
           ]  # global batch size in samples


def get_parameters(yaml_file):
    local_yamls = False if 'https' in yaml_file else True
    if local_yamls:
        # Load the YAML into a parameters dictionary
        with open(yaml_file) as f:
            parameters = yaml.safe_load(f)
    else:
        # Download parameter yaml
        req = requests.get(yaml_file)
        # Load the YAML into a parameters dictionary
        parameters = yaml.safe_load(req.text)

    return parameters


def get_cluster_gpu_types(cluster):
    return [gpu_info[0] for gpu_info in CLUSTER_INFO[cluster]]


def get_gpu_types(clusters):
    gpu_types = set()
    for c in clusters:
        for g in get_cluster_gpu_types(c):
            gpu_types.add(g)
    return gpu_types


def get_gpu_nums(clusters, gpu_types):
    max_gpus_per_run = 1
    for c in clusters:
        for gpu_info in CLUSTER_INFO[c]:
            if gpu_info[0] in gpu_types:
                max_gpus_per_run = max(max_gpus_per_run, gpu_info[1])

    gpu_nums = [1]
    while gpu_nums[-1] < max_gpus_per_run:
        gpu_nums += [2 * gpu_nums[-1]]

    return gpu_nums


def get_valid_gpu_lim(cluster, gpu_type):
    for gpu_info in CLUSTER_INFO[cluster]:
        if gpu_info[0] == gpu_type:
            return gpu_info[1]
    raise ValueError


def mod_parameters(
    parameters,
    max_seq_len,
    global_train_batch_size,
    precision,
    fsdp_config_mixed_precision='DEFAULT',
    run_name='',
    streaming_data=False,
    max_duration='12ba',
    eval_interval='500ba',
    microbatch_size=None,
    wandb=True,
):
    if run_name:
        parameters['run_name'] = run_name
    if streaming_data:
        parameters[
            'data_remote'] = 's3://my-bucket/my-copy-c4'  # TODO: updt if using streaming data
        parameters['train_loader']['dataset']['remote'] = parameters[
            'data_remote']
        parameters['eval_loader']['dataset']['remote'] = parameters[
            'data_remote']

        parameters['data_local'] = '/tmp/c4'
        parameters['train_loader']['dataset']['local'] = parameters[
            'data_local']
        parameters['eval_loader']['dataset']['local'] = parameters['data_local']
    # set max_seq_len
    parameters['max_seq_len'] = max_seq_len
    parameters['model']['max_seq_len'] = max_seq_len
    parameters['tokenizer']['args']['max_seq_len'] = max_seq_len
    parameters['train_loader']['dataset']['max_seq_len'] = max_seq_len
    parameters['eval_loader']['dataset']['max_seq_len'] = max_seq_len

    parameters['global_train_batch_size'] = global_train_batch_size
    if microbatch_size is not None:
        parameters['device_train_microbatch_size'] = microbatch_size

    # update eval batch size based on change in seq len
    parameters['device_eval_batch_size'] = max(
        1,
        int(parameters['device_eval_batch_size'] / ((max_seq_len / 2048)**2)))

    parameters['train_loader']['dataset'][
        'split'] = 'val'  # for throughput testing purposess
    parameters['eval_loader'][
        'eval_subset_num_batches'] = 2  # for throughput testing purposes

    parameters['max_duration'] = max_duration
    parameters['eval_interval'] = eval_interval

    parameters['precision'] = precision
    parameters['fsdp_config']['mixed_precision'] = fsdp_config_mixed_precision

    if wandb:
        # add wandb
        parameters['loggers'] = {'wandb': {}}

    return parameters


def get_integrations(project, wandb=True):
    integrations = [{
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/examples',
        'git_branch': 'main',
        'pip_install': '-r llm/requirements.txt'
    }]
    if wandb:
        integrations += [{
            'integration_type': 'wandb',
            'entity': 'mosaic-ml',
            'project': project
        }]

    return integrations


def run_config(config, args, project, image, RUN):
    yaml_base, model_yaml, max_seq_len, global_train_batch_size, cluster, gpu_type, gpu_num, precision, fsdp_config_mixed_precision = config

    streaming_data = True if 'https' in yaml_base else False
    integrations = get_integrations(
        project,
        wandb=not args.disable_wandb)  # point to git repo and potentially wandb

    # Define our command
    if streaming_data:
        command = """
        composer examples/llm/main.py /mnt/config/parameters.yaml
        """
    else:
        command = """
        python examples/llm/convert_c4.py --out_root ./my-copy-c4 --splits val

        composer examples/llm/main.py /mnt/config/parameters.yaml
        """

    yaml_file = yaml_base + model_yaml
    parameters = get_parameters(yaml_file)

    model_name = '-'.join(yaml_file.split('.')[-2].split('/')[-2:]).replace(
        '_', '-')
    model_name = model_name.split('-')
    if 'mosaic' in model_name:
        model_name.pop(model_name.index('mosaic'))
    model_name = ''.join(model_name)
    name = f"{project}-{cluster}-{model_name}-{gpu_num}x{gpu_type}-s{max_seq_len}b{global_train_batch_size}{precision.replace('amp_', '')}".replace(
        '_', '-')

    name_len_lim = 54 - 7
    if len(name) > name_len_lim:
        _name = name
        name = name[:name_len_lim]
        print(f'Shortening {_name} to {name} ({name_len_lim} chars)')

    microbatch_size = args.microbatch_size if args.microbatch_size else 'auto'
    parameters = mod_parameters(
        parameters,
        max_seq_len,
        global_train_batch_size,
        precision,
        fsdp_config_mixed_precision=fsdp_config_mixed_precision,
        run_name=name,
        streaming_data=streaming_data,
        microbatch_size=microbatch_size,
        wandb=not args.disable_wandb)

    # Create run config mcli sdk/api
    config = RunConfig(
        run_name=name,
        name=name,
        gpu_type=gpu_type,
        gpu_num=gpu_num,
        cpus=None,
        platform=None,
        cluster=cluster,
        image=image,
        optimization_level=0,
        integrations=integrations,
        command=command,
        parameters=parameters,
    )

    if RUN:
        # Create the run from a config
        run = create_run(config)  # , _priority='low'
        print(f'Launching run {run.name}')
    else:
        print(f'run = {name}')


def run_check_capacity(model_yaml, gpu_num, gpu_type, p_multiplier=16):
    _params = model_yaml.replace('.yaml', '')
    params, mult = int(_params[:-1]), _params[-1]
    if mult == 'm':
        b_params = params / 1000
    elif mult == 'b':
        b_params = params
    else:
        raise ValueError

    gpu_mem = int(gpu_type.split('_')[-1][:-2])

    if p_multiplier * b_params > gpu_num * gpu_mem:
        print(
            f'WARNING: will not be running {model_yaml=} on {gpu_num=} {gpu_type=} since it probably will not fit into memory'
        )
        return False
    return True


if __name__ == '__main__':
    args = parse_args()

    n_jobs = 0
    for max_seq_len in get_max_seq_lens(args.seq_len_exp):
        for global_train_batch_size in get_global_train_batch_sizes(
                max_seq_len, args.batch_size_exp):
            for cluster in args.clusters:
                for gpu_type in get_cluster_gpu_types(cluster):
                    ng_lim = get_valid_gpu_lim(cluster, gpu_type)
                    _gpu_nums = [ng for ng in args.gpu_nums if ng <= ng_lim]
                    for gpu_num in _gpu_nums:
                        for precision in args.precisions:
                            for model_yaml in args.model_yamls:

                                run = run_check_capacity(model_yaml,
                                                         gpu_num,
                                                         gpu_type,
                                                         p_multiplier=4)
                                if run:
                                    config = (args.yaml_base, model_yaml,
                                              max_seq_len,
                                              global_train_batch_size, cluster,
                                              gpu_type, gpu_num, precision,
                                              args.fsdp_config_mixed_precision)
                                    print(config)
                                    run_config(config,
                                               args,
                                               project=args.project,
                                               image=args.image,
                                               RUN=args.RUN)
                                    n_jobs += 1

    print(f'{n_jobs=}')
