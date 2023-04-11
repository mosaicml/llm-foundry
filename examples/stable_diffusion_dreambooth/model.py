# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Stable Diffusion ComposerModel."""

from functools import partial
from typing import Optional

import diffusers
import torch
import torch.nn.functional as F
import transformers
from composer.models import ComposerModel
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel)
from diffusers.utils.import_utils import is_xformers_available
from torchmetrics import Metric, MetricCollection
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig


class StableDiffusion(ComposerModel):
    """Stable Diffusion ComposerModel.

    This is a Latent Diffusion model conditioned on text prompts that are run through
    a pre-trained CLIP or LLM model. The CLIP outputs are then passed to as an
    additional input to our Unet during training and can later be used to guide
    the image generation process.

    Args:
        unet (torch.nn.Module): HuggingFace conditional unet, must accept a
            (B, C, H, W) input, (B,) timestep array of noise timesteps,
            and (B, 77, 768) text conditioning vectors.
        vae (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        noise_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the forward diffusion process (training).
        inference_scheduler (diffusers.SchedulerMixin): HuggingFace diffusers
            noise scheduler. Used during the backward diffusion process (inference).
        num_images_per_prompt (int): How many images to generate per prompt
            for evaluation. Default: `1`.
        loss_fn (torch.nn.Module): torch loss function. Default: `F.mse_loss`.
        train_text_encoder (bool): Whether to train the text encoder.
            Default: `False`.
        train_unet (bool): Whether to train the unet. Default: `True`.
        prediction_type (str): `epsilon` or `v_prediction`. `v_prediction` is
            used in parts of the stable diffusion v2.1 training process.
            See https://arxiv.org/pdf/2202.00512.pdf.
            Default: `None` (uses whatever the pretrained model used)
        train_metrics (list): List of torchmetrics to calculate during training.
            Default: `None`.
        val_metrics (list): List of torchmetrics to calculate during validation.
            Default: `None`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
    """

    def __init__(
            self,
            unet: torch.nn.Module,
            vae: torch.nn.Module,
            text_encoder: torch.nn.Module,
            tokenizer: transformers.PreTrainedTokenizer,
            noise_scheduler: diffusers.SchedulerMixin,
            inference_scheduler: diffusers.SchedulerMixin,
            num_images_per_prompt: int = 1,
            loss_fn: torch.nn.Module = F.mse_loss,  # type: ignore
            train_text_encoder: bool = False,
            train_unet: bool = True,
            prediction_type: Optional[str] = None,  # type: ignore
            train_metrics: Optional[list] = None,
            val_metrics: Optional[list] = None,
            image_key: str = 'image_tensor',
            caption_key: str = 'input_ids'):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.inference_scheduler = inference_scheduler

        # let schedulers knows if we're predicting the noise residual or V
        if prediction_type:
            if prediction_type not in ['v_prediction', 'epsilon']:
                raise ValueError(
                    'prediction_type must be "v_prediction" or "epsilon"')
            self.noise_scheduler.config.prediction_type = prediction_type
            self.inference_scheduler.config.prediction_type = prediction_type

        # freeze vae during diffusion training
        self.vae.requires_grad_(False)
        # freeze text_encoder during diffusion training
        if not train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if not train_unet:
            self.unet.requires_grad_(False)

        self.loss_fn = loss_fn

        self.train_metrics = MetricCollection(
            train_metrics) if train_metrics else None
        self.val_metrics = MetricCollection(
            val_metrics) if val_metrics else None

        self.image_key = image_key
        self.caption_key = caption_key
        self.num_images_per_prompt = num_images_per_prompt

    def forward(self, batch):
        inputs, conditioning = batch[self.image_key], batch[self.caption_key]

        # Encode the images to the latent space
        latents = self.vae.encode(
            inputs)['latent_dist'].sample().data  # type: ignore
        # Magical scaling number (See https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
        latents *= 0.18215

        # Encode the text. Assume that the text is already tokenized.
        conditioning = self.text_encoder(conditioning)[0]  # (bs, 77, 768)

        # Sample the diffusion timesteps
        timesteps = torch.randint(0,
                                  len(self.noise_scheduler),
                                  (latents.shape[0],),
                                  device=latents.device)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(latents)

        noised_latents = self.noise_scheduler.add_noise(latents, noise,
                                                        timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise,
                                                       timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.noise_scheduler.config.prediction_type}'
            )
        # Forward through the model
        return self.unet(noised_latents, timesteps,
                         conditioning)['sample'], target

    def loss(self, outputs, batch):
        """Loss between unet output and added noise, typically mse."""
        return self.loss_fn(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        # outputs exist during training, during evaluation we pass the batch to `generate`
        return outputs if outputs else self.generate(batch['prompt'],
                                                     disable_progress_bar=True)

    @torch.no_grad()
    def generate(self,
                 prompt: list,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[list] = None,
                 num_images_per_prompt: Optional[int] = None,
                 disable_progress_bar: Optional[bool] = False,
                 seed: Optional[int] = None):
        """Generates image from noise.

        Performs the backward diffusion process, each inference step takes
        one forward pass through the unet.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide the image generation.
            height (int, optional): The height in pixels of the generated image.
                Default: `self.unet.config.sample_size * 8)`.
            width (int, optional): The width in pixels of the generated image.
                Default: `self.unet.config.sample_size * 8)`.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense
                of slower inference. Default: `50`.
            guidance_scale (float): Guidance scale as defined in
                Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation
                2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
                Higher guidance scale encourages to generate images that are closely linked
                to the text prompt, usually at the expense of lower image quality.
                Default: `7.5`.
            negative_prompt (str or List[str]): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1).
                Must be the same length as list of prompts. Default: `None`.
            num_images_per_prompt (int): The number of images to generate per prompt.
                 Default: `1`.
            disable_progress_bar (bool): Wether to disable the tqdm progress bar during generation.
                Default: `False`.
            seed (int): Random seed to use for generation. Set a seed for reproducible generation.
                Default: `None`.
        """
        if seed:
            torch.manual_seed(seed)
        num_images_per_prompt = num_images_per_prompt if num_images_per_prompt else self.num_images_per_prompt
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        if negative_prompt:
            negative_prompt_bs = 1 if isinstance(negative_prompt,
                                                 str) else len(negative_prompt)
            if negative_prompt_bs != batch_size:
                raise ValueError(
                    f'len(prompts) and len(negative_prompts) must be the same. A negative prompt must be provided for each given prompt.'
                )

        vae_scale_factor = 8
        sample_size = self.unet.config.sample_size  # type: ignore
        height = height or sample_size * vae_scale_factor  # type: ignore
        width = width or sample_size * vae_scale_factor  # type: ignore

        device = self.vae.device
        # tokenize and encode text prompt
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt,
                                               seq_len, -1)

        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            unconditional_input = negative_prompt or ([''] * batch_size)

            # tokenize + encode negative or uncoditional prompt
            unconditional_input = self.tokenizer(
                unconditional_input,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt')
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(device))[0]

            # duplicate unconditional embeddings if we want to generate multiple images per prompt
            bs_embed, seq_len, _ = unconditional_embeddings.shape
            unconditional_embeddings = unconditional_embeddings.repeat(
                1, num_images_per_prompt, 1)
            unconditional_embeddings = unconditional_embeddings.view(
                bs_embed * num_images_per_prompt, seq_len, -1)

            # concat uncond + prompt
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings])

        # prepare for diffusion generation process
        latents = torch.randn(
            (
                batch_size * num_images_per_prompt,
                self.unet.in_channels,  # type: ignore
                height // vae_scale_factor,
                width // vae_scale_factor),
            device=device,  # type: ignore
            dtype=text_embeddings.dtype)  # type: ignore
        self.inference_scheduler.set_timesteps(num_inference_steps)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.inference_scheduler.init_noise_sigma

        # backward diffusion process
        # self.inference_scheduler.timesteps = self.inference_scheduler.timesteps.to(dtype=text_embeddings.dtype) # for mps
        for t in tqdm(self.inference_scheduler.timesteps,
                      disable=disable_progress_bar):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.inference_scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input,
                                   t,
                                   encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t,
                                                    latents).prev_sample

        # We now use the vae to decode the generated latents back into the image.
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents  # type: ignore
        image = self.vae.decode(latents).sample  # type: ignore
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics
        if not metrics:
            return {}
        elif isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric
        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        prompts = batch
        if metric.__class__.__name__ == 'CLIPScore':
            metric.update(outputs, prompts)


def prior_preservation_loss(model_pred: torch.Tensor,
                            target: torch.Tensor,
                            prior_loss_weight: float = 1.0):
    if prior_loss_weight != 1.0:
        model_pred, model_pred_prior = torch.chunk(model_pred, chunks=2, dim=0)
        target, target_prior = torch.chunk(target, chunks=2, dim=0)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        prior_loss = F.mse_loss(model_pred_prior.float(),
                                target_prior.float(),
                                reduction='mean')

        return loss + prior_loss_weight * prior_loss
    return F.mse_loss(model_pred.float(), target.float(), reduction='mean')


def build_stable_diffusion_model(model_name_or_path: str,
                                 train_text_encoder: bool = False,
                                 train_unet: bool = True,
                                 num_images_per_prompt: int = 1,
                                 image_key: str = 'image_tensor',
                                 caption_key: str = 'input_ids',
                                 pretrained: bool = True,
                                 prior_loss_weight: float = 1.0):
    """Builds a Stable Diffusion ComposerModel.

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
        pretrained (bool): If true, download model weights. Default: `False`.
        prior_loss_weight (float): prior preservation loss weight. Default: `1.0`.
    """
    if not pretrained:
        unet = UNet2DConditionModel(**PretrainedConfig.get_config_dict(
            model_name_or_path, subfolder='unet')[0])

        config_dict = PretrainedConfig.get_config_dict(model_name_or_path,
                                                       subfolder='vae')[0]
        # this argument was introduced in a later version of diffusers
        if 'scaling_factor' in config_dict:
            del config_dict['scaling_factor']
        vae = AutoencoderKL(**config_dict)
        text_encoder = CLIPTextModel(config=PretrainedConfig.from_pretrained(
            model_name_or_path, subfolder='text_encoder'))

    else:
        unet = UNet2DConditionModel.from_pretrained(model_name_or_path,
                                                    subfolder='unet')

        vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(model_name_or_path,
                                                     subfolder='text_encoder')

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f'Building without xformers, {e}.')
    noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path,
                                                    subfolder='scheduler')
    inference_scheduler = DDIMScheduler.from_pretrained(model_name_or_path,
                                                        subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path,
                                              subfolder='tokenizer')
    return StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        train_unet=train_unet,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
        train_text_encoder=train_text_encoder,
        num_images_per_prompt=num_images_per_prompt,
        image_key=image_key,
        caption_key=caption_key,
        loss_fn=partial(prior_preservation_loss,
                        prior_loss_weight=prior_loss_weight))  # type: ignore
