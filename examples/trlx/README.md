# TRLX on the Mosaic Platform

[The Mosaic platform](https://www.mosaicml.com/blog/mosaicml-cloud-demo) enables easy training of distributed machine learning (ML) jobs. In this folder, we provide an example of how to run [TRLX](https://github.com/CarperAI/trlx), CarperAI's distributed training library for training large language models with reinforcement learning.

You’ll find in this folder:
- `single_node.yaml` - a yaml to run a single-node TRLX training job on the Mosaic platform.
- `multi_node.yaml` - a yaml to run a multi-node TRLX training job on the Mosaic platform.

## Prerequisites

All you will need to get started is a Docker image (we recommend `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04`)!

## Starting Training
For this example, we will be running the [`simulacra`](https://github.com/CarperAI/trlx/blob/main/examples/simulacra.py) example from TRLX. We include the [MCLI YAML](https://mcli.docs.mosaicml.com/en/latest/main_concepts/yaml_schema.html) configs required to run single or multi-node TRLX on the MosaicML platform. You just need to fill in the `cluster` field in the YAML files. The provided YAMLs use either 8 or 16 GPUs, but all you have to do to use more is change the `gpu_num` field, and modify the `--num_processes` and `--num_machines` arguments to the `accelerate launch` command. See the [TRLX README](https://github.com/CarperAI/trlx/blob/main/README.md) for more information.

************Single-Node Jobs************

Running a single-node job is as simple as running `mcli run -f single_node.yaml`.

There are a lot of logs emitted, but early on you should see something like

```
[2023-03-31 19:04:40,763] [INFO] [config.py:1022:print]   wall_clock_breakdown ......... False
[2023-03-31 19:04:40,763] [INFO] [config.py:1022:print]   world_size ................... 8
[2023-03-31 19:04:40,763] [INFO] [config.py:1022:print]   zero_allow_untested_optimizer  True
```

and then once training has started

```
[RANK 0] Evaluating model
[generation sweep 0/1 | eval batch 0/1]:   0%|          | 0/1 [00:00<?, ?it/s]You're using a GPT2TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
[generation sweep 1/1 | eval batch 1/1]: 100%|██████████| 1/1 [00:01<00:00,  1.16s/it]
[RANK 0] Summarizing evaluation
                                 Evaluation #0
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ prompt                  ┃ output                                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Hatsune Miku, Red Dress │  Daisy to Steal Beans to 460 Beans to 460 Beans to │
│                         │ 460 Beans to 460 Beans to 460 Beans to 460 Beans   │
│                         │ to 460 Beans to 460 Beans to 460 Beans to 460      │
│                         │ Beans to 460 Beans to 460 O O O Beans to 460 O O O │
│                         │ Beans to 365 O O O Beans to                        │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Hatsune Miku, Red Dress │  Queen O O illegcer to O O O O O O O O O O O O O O │
│                         │ O O O O O O O O O O O O O O O O O O O O O O O O O  │
│                         │ O O O O O O O O O O O                              │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Hatsune Miku, Red Dress │  Queen Oisle Queen Oislement Queen Oritwitch O O O │
│                         │ O O O O O O O O O O O O O O O O O O O O O O O O O  │
│                         │ O O O O O O O O O O O O O O O O O                  │
└─────────────────────────┴────────────────────────────────────────────────────┘
[losses/loss: 5.53 | losses/loss_q: 0.59 | losses/loss_v: 0.13 | losses/loss_cql: 18.08 | losses/loss_awac: 3.00]:  10%|▉         | 99/1000 [00:29<04:24,  3.41it/s][RANK 0] Evaluating model
[generation sweep 1/1 | eval batch 1/1]: 100%|██████████| 1/1 [00:00<00:00,  1.95it/s]
[RANK 0] Summarizing evaluation
```


************Multi-Node Jobs************

Running a multi-node job is as simple as running `mcli run -f multi_node.yaml`.

There are a lot of logs emitted, but early on you should see something like

```
[2023-03-31 19:07:11,815] [INFO] [config.py:1022:print]   wall_clock_breakdown ......... False
[2023-03-31 19:07:11,815] [INFO] [config.py:1022:print]   world_size ................... 16
[2023-03-31 19:07:11,815] [INFO] [config.py:1022:print]   zero_allow_untested_optimizer  True
```

and then once training has started

```
[RANK 0] Evaluating model
[generation sweep 0/1 | eval batch 0/1]:   0%|          | 0/1 [00:00<?, ?it/s]You're using a GPT2TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
[generation sweep 1/1 | eval batch 1/1]: 100%|██████████| 1/1 [00:01<00:00,  1.12s/it]
[RANK 0] Summarizing evaluation
                                 Evaluation #0
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ prompt                  ┃ output                                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Hatsune Miku, Red Dress │  Daisy to Steal Beans to 460 Beans to 460 Beans to │
│                         │ 460 Beans to 460 Beans to 460 Beans to 460 Beans   │
│                         │ to 460 Beans to 460 Beans to 460 Beans to 460      │
│                         │ Beans to 460 Beans to 460 O O O Beans to 460 O O O │
│                         │ Beans to 365 O O O Beans to                        │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Hatsune Miku, Red Dress │  Queen O O illegcer to O O O O O O O O O O O O O O │
│                         │ O O O O O O O O O O O O O O O O O O O O O O O O O  │
│                         │ O O O O O O O O O O O                              │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Hatsune Miku, Red Dress │  Queen Oisle Queen Oislement Queen Oritwitch O O O │
│                         │ O O O O O O O O O O O O O O O O O O O O O O O O O  │
│                         │ O O O O O O O O O O O O O O O O O                  │
└─────────────────────────┴────────────────────────────────────────────────────┘
[losses/loss: 4.98 | losses/loss_q: 0.29 | losses/loss_v: 0.07 | losses/loss_cql: 18.88 | losses/loss_awac: 2.72]:  10%|▉         | 99/1000 [00:30<04:36,  3.26it/s][RANK 0] Evaluating model
[generation sweep 1/1 | eval batch 1/1]: 100%|██████████| 1/1 [00:00<00:00,  1.76it/s]
[RANK 0] Summarizing evaluation
```
