# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse

from mcli import sdk as msdk

GPU_AVAILABLE_FLOPS = 312_000_000_000_000


def parse_args():
    parser = argparse.ArgumentParser(description="""
        Parse run configs to get MosaicGPT training throughput.
        MFU and HFU are defined in https://arxiv.org/abs/2205.05198
        All FLOP calculations do not include norm, act, residual, etc.
        """)

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument('--filters', type=str, default=[], nargs='+')

    return parser.parse_args()


def extract_from_loglines(string, lines):
    for line in lines:
        if f'{string}: ' in line[:len(f'{string}: ')]:
            return line.split(' ')[-1]

    raise ValueError(f'{string=} not found in log')


def get_runs(args):
    runs = [r for r in msdk.get_runs() if args.project in r.name]
    for filter in args.filters:
        runs = [r for r in runs if filter in r.name]

    def sort_key(r):
        model_name = [s for s in r.name.split('-') if 'gpt' in s][0]
        num_gpu = r.config.gpu_num
        if model_name[-1] == 'm':
            model_name_size = 1e6
        elif model_name[-1] == 'b':
            model_name_size = 1e9
        else:
            print(model_name)
            raise ValueError
        model_size = int(model_name[3:-1])
        return (model_name_size, model_size, r.config.parameters['max_seq_len'],
                num_gpu, r.config.parameters['global_train_batch_size'])

    runs.sort(reverse=True, key=sort_key)

    return runs


def filter_runs(runs):
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
        if run.status in [
                msdk.RunStatus('FAILED_PULL'),
                msdk.RunStatus('PENDING'),
                msdk.RunStatus('QUEUED'),
                msdk.RunStatus('RUNNING'),
                msdk.RunStatus('SCHEDULED'),
                msdk.RunStatus('STARTING'),
                msdk.RunStatus('STOPPED'),
                msdk.RunStatus('STOPPING'),
                msdk.RunStatus('TERMINATING'),
        ]:
            print(f'run {run.name} has run status {run.status}')
            pop_runs.append(run)
    for run in pop_runs:
        runs.pop(runs.index(run))

    return runs


def parse_run(run):
    n_params = gpu_num = seq_len = global_batchsize_tokens = mfu = -1
    global_train_batch_size = micro_batchsize = precision = throughput = -1

    model_name = [s for s in run.name.split('-') if 'gpt' in s][0]
    gpu_num = run.config.gpu_num
    gpu_type = run.config.gpu_type

    precision = run.config.parameters['precision']
    mp_mode = run.config.parameters['fsdp_config']['mixed_precision']
    sharding_strategy = run.config.parameters['fsdp_config'][
        'sharding_strategy']
    a_ckpt = run.config.parameters['fsdp_config']['activation_checkpointing']
    a_cpu_offload = run.config.parameters['fsdp_config'][
        'activation_cpu_offload']
    seq_len = run.config.parameters['max_seq_len']
    global_train_batch_size = run.config.parameters['global_train_batch_size']

    logs = msdk.get_run_logs(run)
    lines = ''
    for line in logs:
        lines += line
    lines = lines.split('\n')

    for line in lines:
        if f'n_params: ' in line[:len(f'n_params: ')]:
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

    d_model = run.config.parameters['model']['d_model']
    n_layers = run.config.parameters['model']['n_layers']

    global_batchsize_tokens = global_train_batch_size * seq_len
    grad_accum = global_train_batch_size // gpu_num // micro_batchsize

    throughput_t = throughput * seq_len
    throughput_t_gpu = throughput_t / gpu_num

    # mfu is approximated using thoughtput and param count
    # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
    # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
    # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
    # this gets us FLOPs / token
    flops_per_token = 2 * n_params
    flops_per_seq = flops_per_token * seq_len
    mfu = 3 * flops_per_seq * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)

    # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
    attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))
    # there are 2 ops in bwd pass and 1 in fwd pass so we mult by 3
    mfu_w_attn = (3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / (
        gpu_num * GPU_AVAILABLE_FLOPS)

    if a_ckpt:
        hfu = 4 * flops_per_seq * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
        hfu_w_attn = (4 * flops_per_seq + 4 * attn_flops_per_seq
                     ) * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
    else:
        hfu = mfu
        hfu_w_attn = mfu_w_attn

    print(
        f'| {model_name: >7} | {seq_len: >10} | {gpu_num: >3}x{gpu_type.upper()} | {mfu:.4f} | {mfu_w_attn:.4f} | {hfu:.4f} | {hfu_w_attn:.4f} | {throughput_t: >16.2f} | {throughput_t_gpu: >23.4f} | {throughput: >16.4f} | {n_params: >11} | {global_batchsize_tokens: >19} | {global_train_batch_size: >19} | {micro_batchsize: >18} | {grad_accum: >9} | {sharding_strategy: >13} | {str(a_ckpt): >7} | {str(a_cpu_offload): >13} | {precision: >9} | {mp_mode: >7} | {gpu_num: >7} | {gpu_type: >7} |'
    )


def main(args):
    runs = get_runs(args)
    runs = filter_runs(runs)

    print(
        '| Model   | SeqLen (T) | GPUType       | MFU*   | MFU    | HFU*   | HFU    | Throughput (T/s) | GPUThroughput (T/s/GPU) | Throughput (S/s) | ParamCount  | GlobalBatchSize (T) | GlobalBatchSize (S) | MicroBatchSize (S) | GradAccum | ShardStrategy | ActCkpt | ActCPUoffload | Precision | MP Mode | NumGPUs | GPUType   |\n'
        '| ------- | ---------- | ------------- | ------ | ------ | ------ | ------ | ---------------- | ----------------------- | ---------------- | ----------- | ------------------- | ------------------- | ------------------ | --------- | ------------- | ------- | ------------- | --------- | ------- | ------- | --------- |'
    )
    for run in runs:
        try:
            parse_run(run)
        except Exception as e:
            print(f'{run.name=} not parsed')
            print(e)


if __name__ == '__main__':
    args = parse_args()
    main(args)

    from mcli.api.engine.engine import MAPIConnection
    MAPIConnection.get_current_connection().close()
