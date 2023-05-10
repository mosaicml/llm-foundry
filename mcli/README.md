# Using MosaicML platform

This folder contains examples of how to use [MosaicML platform](https://www.mosaicml.com/platform) to launch LLM workloads.

Full documentation on MosaicML platform can be found at https://mcli.docs.mosaicml.com/.

## Using MosaicML Command Line Interface (MCLI) to launch runs

In this folder, we provide two MCLI examples, [`mcli-1b.yaml`](./mcli-1b.yaml) and [`mcli-1b-custom.yaml`](./mcli-1b-custom.yaml) that you can use to launch training runs using our command-line tool, `mcli`. We also include an MCLI example, [`pretokenize_oci_upload.yaml`](./pretokenize_oci_upload.yaml) which pre-tokenizes the C4 dataset and uploads it to a desired bucket in Oracle Cloud Infrastructure (OCI).

The first example, `mcli-1b.yaml` describe a job to train an MPT-1B model. The job simply clones this repo, checks out a particular tag, and runs the `scripts/train/train.py` training script. The workload config is read from a YAML in this repo: [`yamls/pretrain/mpt-1b.yaml`](../scripts/train/yamls/pretrain/mpt-1b.yaml).

The second example, `mcli-1b-custom.yaml`, shows how to inject a custom config at runtime (`/mnt/config/parameters.yaml`) and pass that file to the `scripts/train/train.py` training script. This workflow allows you to quickly customize a training run without needing to commit and push changes to the repository.

Here's how easy it is to launch an LLM training run with MCLI:

<!--pytest.mark.skip-->
```bash
mcli run -f mcli-1b.yaml --cluster CLUSTER --gpus GPUS --name NAME --follow
```

All the details of multi-gpu and multi-node orchestration are handled automatically by MosaicML platform. Try it out yourself!

## Using the MosaicML Python SDK to launch runs
You can also use the [Python SDK](https://mcli.docs.mosaicml.com/en/stable/python/hello_world.html) to launch MosaicML platform jobs.
This can be used to programatically sweep hyperparameters or orchestrate training runs within a larger pipeline.
