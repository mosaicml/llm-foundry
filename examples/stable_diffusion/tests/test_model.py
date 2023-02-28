# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import (AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler,
                       UNet2DConditionModel)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

from examples.stable_diffusion.model import StableDiffusion


def build_dummy_stable_diffusion_model(model_name_or_path: str,
                                       train_text_encoder: bool = False,
                                       train_unet: bool = True,
                                       num_images_per_prompt: int = 1,
                                       image_key: str = 'image_tensor',
                                       caption_key: str = 'input_ids'):
    """Create a stable diffusion model without the weights.

    Args:
        model_name_or_path (str): Path to a pretrained HuggingFace model or config.
            Commonly "CompVis/stable-diffusion-v1-4" or "stabilityai/stable-diffusion-2-1".
        train_text_encoder (bool): Whether to train the text encoder. Default: `False`.
        train_unet (bool): Whether to train the unet. Default: `True`.
        num_images_per_prompt (int): How many images to generate per prompt for evaluation.
            Default: `1`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
    """
    unet = UNet2DConditionModel(**PretrainedConfig.get_config_dict(
        model_name_or_path, subfolder='unet')[0])
    if torch.cuda.is_available() and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    config_dict = PretrainedConfig.get_config_dict(model_name_or_path,
                                                   subfolder='vae')[0]
    # this argument was introduced in a later version of diffusers
    if 'scaling_factor' in config_dict:
        del config_dict['scaling_factor']
    vae = AutoencoderKL(**config_dict)

    text_encoder = CLIPTextModel(config=PretrainedConfig.from_pretrained(
        model_name_or_path, subfolder='text_encoder'))
    noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path,
                                                    subfolder='scheduler')

    # less parameters than DDIM and good results.
    # see https://arxiv.org/abs/2206.00364 for information on choosing inference schedulers.
    inference_scheduler = LMSDiscreteScheduler.from_pretrained(
        model_name_or_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path,
                                              subfolder='tokenizer')
    return StableDiffusion(unet=unet,
                           vae=vae,
                           text_encoder=text_encoder,
                           train_unet=train_unet,
                           tokenizer=tokenizer,
                           noise_scheduler=noise_scheduler,
                           inference_scheduler=inference_scheduler,
                           train_text_encoder=train_text_encoder,
                           num_images_per_prompt=num_images_per_prompt,
                           image_key=image_key,
                           caption_key=caption_key)


@pytest.mark.parametrize(
    'model_name',
    ['CompVis/stable-diffusion-v1-4', 'stabilityai/stable-diffusion-2-1'])
def test_model_builder(model_name):
    # test that the StableDiffusion base class outputs the correct size outputs
    # for both model versions.
    model = build_dummy_stable_diffusion_model(model_name)
    batch_size = 1
    H = 64
    W = 64
    image = torch.randn(batch_size, 3, H, W)
    latent = torch.randn(batch_size, 4, H // 8, W // 8)
    caption = torch.randint(low=0,
                            high=128,
                            size=(
                                batch_size,
                                77,
                            ),
                            dtype=torch.long)

    batch = {'image_tensor': image, 'input_ids': caption}
    output, target = model(
        batch
    )  # model.forward generates the unet output noise or v_pred target.
    assert output.shape == latent.shape
    assert target.shape == latent.shape
