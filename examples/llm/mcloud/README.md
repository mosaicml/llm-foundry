# Using MosaicML Cloud to train LLMs

This folder contains examples of how to use [MosaicML Cloud](https://www.mosaicml.com/cloud) to launch LLM training runs.

Full documentation on MosaicML Cloud can be found at https://mcli.docs.mosaicml.com/.

## Using MosaicML Command Line Interface (MCLI) to launch runs

In this folder, we provide two MCLI examples, [`mcli-1b.yaml`](./mcli-1b.yaml) and [`mcli-1b-custom.yaml`](./mcli-1b-custom.yaml) that you can use to launch training runs using our command-line tool, `mcli`.

The first example, `mcli-1b.yaml`, simply clones this repo, checks out a particular tag, and runs the `main.py` training script. The workload config is read from a YAML in this repo: [`yamls/mosaic_gpt/1b.yaml`](../yamls/mosaic_gpt/1b.yaml).

The second example, `mcli-1b-custom.yaml`, shows how to inject a custom config at runtime (`/mnt/config/parameters.yaml`) and pass that file to the `main.py` training script. This workflow allows you to quickly customize a training run without needing to commit and push changes to the repository.

Here's how easy it is to launch an LLM training run with MCLI:
```bash
mcli run -f mcli-1b.yaml --cluster CLUSTER --gpus GPUS --name NAME --follow
```

All the details of multi-gpu and multi-node orchestration are handled automatically by MosaicML Cloud. Try it out yourself!

## Using the MosaicML Python SDK to launch runs
You can also use the [Python SDK](https://mcli.docs.mosaicml.com/en/stable/python/hello_world.html) to launch MosaicML Cloud jobs.
This can be used to programatically sweep hyperparameters or orchestrate training runs within a larger pipeline.
