# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import math
from typing import Any, Dict, List, Union

from composer.callbacks.speed_monitor import \
    GPU_AVAILABLE_FLOPS as GPU_FLOP_DICT

from mcli import sdk as msdk


def str_to_bool(value: Union[bool, str]):
    # helper fn
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser(description="""
        Parse run configs to get MPT training throughput.
        MFU and HFU are defined in https://arxiv.org/abs/2205.05198
        All FLOP calculations do not include norm, act, residual, etc.
        """)

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument('--filters', type=str, default=[], nargs='+')
    parser.add_argument('-s',
                        '--save-path',
                        type=str,
                        default='benchmark_results')
    parser.add_argument('-p',
                        '--print-results',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)

    return parser.parse_args()


def get_runs(args: argparse.Namespace):
    runs = [
        r for r in msdk.get_runs(include_details=True)
        if args.project in r.name.split('-')[0] and
        r.status == msdk.RunStatus('COMPLETED')
    ]
    for filter in args.filters:
        runs = [r for r in runs if filter in r.name]

    def sort_key(r: msdk.Run):
        model_name = r.name.split('-')[2]
        num_gpu = r.gpus
        gpu_type = r.gpu_type
        model_precision = r.submitted_config.parameters['precision']
        if model_name[-1] == 'm':
            model_name_size = 1e6
        elif model_name[-1] == 'b':
            model_name_size = 1e9
        else:
            print(model_name)
            raise ValueError
        model_size = int(model_name[:-1])
        return (gpu_type, model_precision, model_name_size, model_size,
                r.submitted_config.parameters['max_seq_len'], num_gpu,
                r.submitted_config.parameters['global_train_batch_size'])

    unique_runs = {sort_key(i): i for i in runs}
    runs = [unique_runs[r] for r in unique_runs]
    runs.sort(reverse=True, key=sort_key)

    return runs


def filter_runs(runs: List[msdk.Run]):
    pop_runs = []
    for run in runs:
        if run.status == msdk.RunStatus('FAILED'):
            print(
                f"run {run.name} has FAILED (likely due to OOM error but we'd recommend checking.)"
            )
            pop_runs.append(run)

    for run in pop_runs:
        runs.pop(runs.index(run))

    pop_runs = []
    for run in runs:
        if run.status != msdk.RunStatus('COMPLETED'):
            print(f'run {run.name} has run status {run.status}')
            pop_runs.append(run)
    for run in pop_runs:
        runs.pop(runs.index(run))

    return runs


def parse_run(run: msdk.Run) -> Dict[str, Any]:
    n_params = micro_batchsize = throughput = -1

    model_name = run.name.split('-')[2]
    gpus = run.gpus
    gpu_type = run.gpu_type

    if 'h100' in gpu_type:
        gpu_type = 'h100-sxm'
    if 'a100' in gpu_type:
        gpu_type = 'a100'
    GPU_AVAILABLE_FLOPS = GPU_FLOP_DICT[gpu_type][
        run.submitted_config.parameters['precision']]

    gpu_type = run.gpu_type
    fsdp_config = run.submitted_config.parameters['fsdp_config']

    seq_len = run.submitted_config.parameters['max_seq_len']
    global_train_batch_size = run.submitted_config.parameters[
        'global_train_batch_size']
    activation_checkpointing = fsdp_config['activation_checkpointing']

    logs = msdk.get_run_logs(run)
    lines = ''
    for line in logs:
        lines += line
    lines = lines.split('\n')

    for line in lines:
        if line.startswith('n_params'):
            n_params = int(line.split(' ')[-1])
            break

    lines.reverse()

    for line in lines:
        if 'trainer/device_train_microbatch_size' in line:
            micro_batchsize = int(line.split(' ')[-1])
            break

    for line in lines:
        if 'throughput/samples_per_sec' in line:
            throughput = float(line.split(' ')[-1])
            break

    d_model = run.submitted_config.parameters['model']['d_model']
    n_layers = run.submitted_config.parameters['model']['n_layers']

    # mfu is approximated using throughput and param count
    # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
    # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
    # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
    # this gets us FLOPs / token
    flops_per_token = 2 * n_params
    flops_per_seq = flops_per_token * seq_len

    # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
    attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))
    # there are 2 ops in bwd pass and 1 in fwd pass so we mult by 3
    mfu_w_attn = (3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / (
        gpus * GPU_AVAILABLE_FLOPS)

    if activation_checkpointing:
        hfu_w_attn = (4 * flops_per_seq + 4 * attn_flops_per_seq
                     ) * throughput / (gpus * GPU_AVAILABLE_FLOPS)
    else:
        hfu_w_attn = mfu_w_attn

    model_tflop = int(
        (3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / gpus / 1e12)

    return {
        'Model':
            model_name,
        'SeqLen (T)':
            seq_len,
        '# GPUs':
            gpus,
        'GPU':
            gpu_type,
        'MFU':
            round(mfu_w_attn * 100, 2),
        'HFU':
            round(hfu_w_attn * 100, 2),
        'Model TFLOP':
            model_tflop,
        'MicroBatchSize':
            micro_batchsize,
        'GradAccum':
            math.ceil(global_train_batch_size / gpus / micro_batchsize),
        'GlobalBatchSize':
            global_train_batch_size,
        'Throughput (S/s)':
            int(throughput),
        'Throughput (T/s)':
            int(throughput * seq_len),
        'Throughput (T/s/GPU)':
            int(throughput * seq_len / gpus),
        'GlobalBatchSize (T)':
            global_train_batch_size * seq_len,
        'Precision':
            run.submitted_config.parameters['precision'],
        'MP Mode':
            fsdp_config['mixed_precision'],
        'Sharding Strategy':
            fsdp_config['sharding_strategy'],
        'Activation Checkpointing':
            activation_checkpointing,
        'Activation CPUOffload':
            str(fsdp_config['activation_cpu_offload']),
        'NumParams':
            n_params,
    }


def main(args: argparse.Namespace):
    runs = get_runs(args)
    runs = filter_runs(runs)

    results = []
    for run in runs:
        try:
            results.append(parse_run(run))
        except Exception as e:
            print(f'{run.name=} not parsed')
            print(e)

    if results:
        csv_name = args.save_path + '.csv'
        with open(csv_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        md_name = args.save_path + '.md'
        fieldnames = results[0].keys()
        with open(md_name, 'w') as f:
            fmt = '| ' + ' {} |' * len(fieldnames) + '\n'

            f.write(fmt.format(*fieldnames))
            f.write(fmt.format(*['---' for _ in fieldnames]))
            if args.print_results:
                print(fmt.format(*fieldnames), end='')
                print(fmt.format(*['---' for _ in fieldnames]), end='')
            for result in results:
                if args.print_results:
                    print(fmt.format(*result.values()), end='')
                f.write(fmt.format(*result.values()))
    else:
        print('WARNING: No results parsed.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
