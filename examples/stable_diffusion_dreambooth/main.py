# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to finetune a Stable Diffusion Model."""

import os
import sys
from pathlib import Path

import torch
from callbacks import LogDiffusionImages, SaveClassImages
from composer import Trainer
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import ConstantScheduler
from composer.utils import dist, reproducibility
from data import build_dreambooth_dataloader, build_prompt_dataloader
from model import build_stable_diffusion_model
from omegaconf import DictConfig, OmegaConf


def main(config: DictConfig):  # type: ignore
    reproducibility.seed_all(config.seed)
    # initialize the pytorch distributed process group if training on multiple gpus.

    if dist.get_world_size() != 0 and config.device == 'gpu':
        dist.initialize_dist(config.device)
        config.train_device_batch_size = config.global_train_batch_size // dist.get_world_size(
        )
        config.eval_device_batch_size = config.global_eval_batch_size // dist.get_world_size(
        )

    else:
        config.train_device_batch_size = config.global_train_batch_size
        config.eval_device_batch_size = config.global_eval_batch_size

    print('Building Composer model')
    model = build_stable_diffusion_model(
        model_name_or_path=config.model.name,
        train_text_encoder=config.model.train_text_encoder,
        train_unet=config.model.train_unet,
        num_images_per_prompt=config.model.num_images_per_prompt,
        image_key=config.model.image_key,
        caption_key=config.model.caption_key,
        prior_loss_weight=config.get('prior_loss_weight', 1.0))

    if config.use_prior_preservation:
        print('generating class images for prior preservation')
        class_images_dir = Path(config.dataset.class_data_root)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            # duplicate the class prompt * num class samples to generate class images
            images_to_generate = config.num_class_images - cur_class_images
            class_prompts = [config.dataset.class_prompt] * images_to_generate
            prompt_dataloader = build_prompt_dataloader(
                class_prompts, batch_size=config.train_device_batch_size)
            save_class_images = SaveClassImages(
                class_data_root=config.dataset.class_data_root)

            model.num_images_per_prompt = 1  # set for prior preservation image generation
            trainer = Trainer(model=model,
                              eval_dataloader=prompt_dataloader,
                              callbacks=save_class_images)
            # eval run will save images via the SaveClassImages callback
            trainer.eval()
            model.num_images_per_prompt = config.model.num_images_per_prompt

    # Train dataset
    print('Building dataloaders')
    train_dataloader = build_dreambooth_dataloader(
        instance_data_root=config.dataset.instance_data_root,
        instance_prompt=config.dataset.instance_prompt,
        use_prior_preservation=config.use_prior_preservation,
        class_data_root=config.dataset.get('class_data_root'),
        class_prompt=config.dataset.get('class_prompt'),
        resolution=config.dataset.resolution,
        center_crop=config.dataset.center_crop,
        tokenizer=model.tokenizer,
        batch_size=config.train_device_batch_size,
        dataloader_kwargs=config.dataset.dataloader_kwargs)

    # Optimizer
    print('Building optimizer and learning rate scheduler')
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.optimizer.lr,
                                  weight_decay=config.optimizer.weight_decay)

    # Constant LR for fine-tuning
    lr_scheduler = ConstantScheduler()

    callbacks = []

    print('Building Speed, LR, and Memory monitoring callbacks')
    # Measures throughput as samples/sec and tracks total training time
    callbacks += [SpeedMonitor(window_size=50)]
    callbacks += [LRMonitor()]  # Logs the learning rate
    callbacks += [MemoryMonitor()]  # Logs memory utilization

    eval_dataloader = None  # we only need an eval dataloader if using w&b logging
    loggers = None
    if config.get('wandb'):
        import wandb
        if wandb.run:
            wandb.config.update(OmegaConf.to_container(config, resolve=True))

        loggers = WandBLogger(**config.wandb)
        callbacks += [LogDiffusionImages()]

        eval_dataloader = build_prompt_dataloader(
            config.dataset.eval_prompts,
            batch_size=config.eval_device_batch_size)

    # Create the Trainer!
    print('Building Trainer')

    fsdp_config = None
    if config.use_fsdp:
        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
            'min_params': 1e9,
            'cpu_offload': False,  # Not supported yet
            'mixed_precision': 'DEFAULT',
            'backward_prefetch': 'BACKWARD_POST',
            'activation_checkpointing': False,
            'activation_cpu_offload': False,
            'verbose': True
        }

    trainer = Trainer(
        run_name=config.run_name,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        loggers=loggers,
        max_duration=config.max_duration,
        eval_interval=config.get('eval_interval', 1),
        callbacks=callbacks,
        save_folder=config.save_folder,
        save_interval=config.save_interval,
        save_num_checkpoints_to_keep=config.save_num_checkpoints_to_keep,
        save_filename=config.save_filename,
        save_weights_only=config.save_weights_only,
        load_path=config.load_path,
        device=config.device,
        precision=config.get('precision'),
        grad_accum=config.get('grad_accum'),
        seed=config.seed,
        fsdp_config=fsdp_config)

    print(OmegaConf.to_yaml(config))

    print('Train!')
    trainer.fit()


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        raise ValueError('The first argument must be a path to a yaml config.')

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = OmegaConf.load(f)
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(yaml_config, cli_config)
    main(config)  # type: ignore
