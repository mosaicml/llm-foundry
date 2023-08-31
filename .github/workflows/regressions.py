def get_configs():
    # regression tests
    # mpt-125m to chinchilla, 8 gpus
    # eval mpt-7b from huggingface hub, 8 gpus
    # eval mpt-7b from composer checkpoint, 8 gpus
    # mpt-125m and convert to huggingface format, 8 gpus
    # finetune llama2-7b for a few steps, 8 gpus
    # mpt-125m sharded with resumption, 16 gpus
    # mpt-125m sharded with resumption, 8 gpus
    run_configs = []

    return run_configs, []

if __name__ == "__main__":
    run_configs, _ = get_configs()
    for run_config in run_configs:
        run_config.run()