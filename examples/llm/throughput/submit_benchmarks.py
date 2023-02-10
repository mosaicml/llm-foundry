# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

import requests
import yaml
from mcli.sdk import RunConfig, create_run, get_clusters


def _get_cluster_info():
    clusters = get_clusters()
    cluster_info = {}

    for cluster in clusters:
        cluster_info[cluster.name] = [(ci.gpu_type.value, max(ci.gpu_nums))
                                      for ci in cluster.cluster_instances
                                      if ci.gpu_type.value is not None]

    return cluster_info


CLUSTER_INFO = _get_cluster_info()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Generate and run configurations to test MosaicGPT training throughput on Mosaic Cloud.'
    )

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument(
        '--image',
        type=str,
        default='mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04')
    parser.add_argument('--git_branch',
                        type=str,
                        default='main',
                        help='what git branch to use.')
    parser.add_argument('--git_commit',
                        type=str,
                        default=None,
                        help='what git commit to use.')
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
    parser.add_argument('--fsdp_config_activation_checkpointing',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=None)
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
        'https://raw.githubusercontent.com/mosaicml/examples/main/examples/llm/yamls/mosaic_gpt/'
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

    parser.add_argument('--pad_vocab_multiple', type=int, default=None)

    parser.add_argument('--data_remote',
                        type=str,
                        default=None,
                        help='optional data remote path for streaming data')

    parser.add_argument('--wandb',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=True)

    parser.add_argument('--RUN',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)

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


def mod_parameters(parameters,
                   max_seq_len,
                   global_train_batch_size,
                   precision,
                   fsdp_config_mixed_precision='DEFAULT',
                   fsdp_config_activation_checkpointing=None,
                   run_name='',
                   data_remote=None,
                   max_duration='30ba',
                   eval_interval=0,
                   microbatch_size=None,
                   wandb=True,
                   pad_vocab_multiple=None):
    if run_name:
        parameters['run_name'] = run_name
    if data_remote is not None:
        parameters['data_remote'] = data_remote
        parameters['train_loader']['dataset']['remote'] = parameters[
            'data_remote']
        parameters['eval_loader']['dataset']['remote'] = parameters[
            'data_remote']

        parameters['data_local'] = '/tmp/c4'
        parameters['train_loader']['dataset']['local'] = parameters[
            'data_local']
        parameters['eval_loader']['dataset']['local'] = parameters['data_local']
    else:
        parameters['train_loader']['dataset'][
            'split'] = 'train_small'  # for throughput testing purposes
    # set max_seq_len
    parameters['max_seq_len'] = max_seq_len
    parameters['model']['max_seq_len'] = max_seq_len

    # Pad vocab size to multiple of N for A100 perf
    if pad_vocab_multiple:
        vocab_size = parameters['model']['vocab_size']
        parameters['model']['vocab_size'] = math.ceil(
            vocab_size / pad_vocab_multiple) * pad_vocab_multiple

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

    parameters['eval_loader'][
        'eval_subset_num_batches'] = 2  # for throughput testing purposes

    parameters['max_duration'] = max_duration
    parameters['eval_interval'] = eval_interval

    parameters['precision'] = precision
    parameters['fsdp_config']['mixed_precision'] = fsdp_config_mixed_precision
    if fsdp_config_activation_checkpointing is not None:
        parameters['fsdp_config'][
            'activation_checkpointing'] = fsdp_config_activation_checkpointing

    if wandb:
        # add wandb
        parameters['loggers'] = {'wandb': {}}

    return parameters


def get_integrations(project, git_branch=None, git_commit=None, wandb=True):
    integrations = []

    if args.git_branch and args.git_commit:
        raise ValueError(
            f'{args.git_branch=} and {args.git_commit=} cannot both be set!')
    git_integration = {
        k: v for k, v in {
            'git_branch': args.git_branch,
            'git_commit': git_commit,
        }.items() if v is not None
    }
    git_integration.update({
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/examples',
        'pip_install': '-e .[llm]'
    })

    integrations = [git_integration]

    if wandb:
        integrations += [{
            'integration_type': 'wandb',
            'entity': 'mosaic-ml',
            'project': args.project
        }]

    return integrations


def run_config(config, args):
    yaml_base, model_yaml, max_seq_len, global_train_batch_size, cluster, gpu_type, gpu_num, precision = config

    integrations = get_integrations(
        args.project,
        git_branch=args.git_branch,
        git_commit=args.git_commit,
        wandb=args.wandb)  # point to git repo and potentially wandb

    # Define our command
    if args.data_remote is not None:
        command = """
        cd examples

        composer examples/llm/main.py /mnt/config/parameters.yaml
        """
    else:
        command = """
        cd examples

        python examples/common/convert_c4.py --out_root ./my-copy-c4 --splits train_small val

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
    name = f"{args.project}-{cluster}-{model_name}-{gpu_num}x{gpu_type}-s{max_seq_len}b{global_train_batch_size}{precision.replace('amp_', '')}".replace(
        '_', '-')

    name_len_lim = 54 - 7
    if len(name) > name_len_lim:
        _name = name
        name = name[:name_len_lim]
        print(f'Shortening {_name} to {name} ({name_len_lim} chars)')

    microbatch_size = args.microbatch_size or 'auto'
    parameters = mod_parameters(
        parameters,
        max_seq_len,
        global_train_batch_size,
        precision,
        fsdp_config_mixed_precision=args.fsdp_config_mixed_precision,
        fsdp_config_activation_checkpointing=args.
        fsdp_config_activation_checkpointing,
        run_name=name,
        data_remote=args.data_remote,
        microbatch_size=microbatch_size,
        wandb=args.wandb,
        pad_vocab_multiple=args.pad_vocab_multiple)

    # Create run config mcli sdk/api
    config = RunConfig(
        run_name=name,
        name=name,
        gpu_type=gpu_type,
        gpu_num=gpu_num,
        cpus=None,
        platform=None,
        cluster=cluster,
        image=args.image,
        optimization_level=0,
        integrations=integrations,
        command=command,
        parameters=parameters,
    )

    if args.RUN:
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
                                              gpu_type, gpu_num, precision)
                                    print(config)
                                    run_config(config, args)
                                    n_jobs += 1

    print(f'{n_jobs=}')
