# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Image captioning dataset creation tools and preprocessing."""

from functools import partial
from pathlib import Path
from typing import Optional

import torch
import transformers
from composer.core import DataSpec
from composer.utils import dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def dreambooth_collate_fn(examples: dict, use_prior_preservation: bool = False):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    image_tensor = [example['instance_images'] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if use_prior_preservation:
        input_ids += [example['class_prompt_ids'] for example in examples]
        image_tensor += [example['class_images'] for example in examples]

    image_tensor = torch.stack(image_tensor)
    image_tensor = image_tensor.to(  # type: ignore
        memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        'input_ids': input_ids,
        'image_tensor': image_tensor,
    }
    return batch


def build_dreambooth_dataloader(instance_data_root: str,
                                instance_prompt: str,
                                tokenizer: transformers.PreTrainedTokenizer,
                                resolution: int,
                                use_prior_preservation: bool = False,
                                center_crop: Optional[bool] = False,
                                class_prompt: Optional[str] = None,
                                class_data_root: Optional[str] = None,
                                *,
                                batch_size: int,
                                drop_last: Optional[bool] = True,
                                shuffle: Optional[bool] = True,
                                **dataloader_kwargs: dict):
    """Builds a dreambooth dataloader.

    Includes transformations and tokenization for diffusion training.

    Args:
        instance_data_root (str): Path to directory of instance images.
        instance_prompt (str): The prompt to associate with instance images.
            Normally in the form <INSTANCE TOKEN> <CLASS>.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for the `text_encoder`
            of the model you are training. For a `CLIPTextModel`  this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        resolution (int): Final image size after processing.
        center_crop (bool): Whether to center crop the images during preprocessing.
            If `False`, `RandomCrop` will be used. Default: `True`.
        use_prior_preservation (bool): Whether to use prior preservation images. Default: `False`.
        class_prompt (str): Prompt associate with prior presevation images. Default: `None`.
        class_data_root (str): Path the image generated from the model for prior preservation.
            Default: `None`.
        batch_size (int): Batch size per device.
        drop_last (bool): Whether to drop last samples. Default: ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the
            dataloader (e.g. num_workers, etc.)
    """
    image_transforms = transforms.Compose([
        transforms.Resize(resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution)
        if center_crop else transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    if not use_prior_preservation:
        class_data_root = None
        class_prompt = None
    dataset = DreamBoothDataset(instance_data_root=instance_data_root,
                                instance_prompt=instance_prompt,
                                class_data_root=class_data_root,
                                class_prompt=class_prompt,
                                tokenizer=tokenizer,
                                image_transforms=image_transforms)
    sampler = dist.get_sampler(
        dataset,
        drop_last=drop_last,  # type: ignore
        shuffle=shuffle)  # type: ignore
    use_prior_preservation = True if class_prompt and class_data_root else False
    collate_fn = partial(dreambooth_collate_fn,
                         use_prior_preservation=use_prior_preservation)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,  # type: ignore
        collate_fn=collate_fn,  # type: ignore
        **dataloader_kwargs)


def build_prompt_dataloader(prompts: list[str], batch_size: int,
                            **dataloader_kwargs):
    """Builds a prompt dataset from a list of strings for eval.

    Args:
        prompts (list[str]): A list of prompts.
        batch_size (int): Batch size per device.
    """
    dataset = PromptDataset(prompts)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)
    return DataSpec(dataloader=DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          sampler=sampler,
                                          drop_last=False,
                                          **dataloader_kwargs),
                    get_num_samples_in_batch=lambda x: len(x['prompt']))


class PromptDataset(Dataset):
    """A simple Dataloader that iterates through a list of strings.

    Args:
        prompts (list[str]): A list of prompts.
    """

    def __init__(self, prompts: list):
        self.prompts = prompts

    def __getitem__(self, index: int):
        example = {}
        example['prompt'] = self.prompts[index]
        example['index'] = index
        return example

    def __len__(self):
        return len(self.prompts)


class DreamBoothDataset(Dataset):
    """Dreambooth Dataset.

    A dataset to prepare the instance and class images with the prompts for
    Dreambooth fine-tuning.

    Args:
        instance_data_root (str): Path to directory of instance images.
        instance_prompt (str): The prompt to associate with instance images.
            Normally in the form <INSTANCE TOKEN> <CLASS>.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for the `text_encoder`
            of the model you are training. For a `CLIPTextModel`  this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        class_prompt (str): Prompt associate with prior presevation images. Default: `None`.
        class_data_root (str): Path the image generated from the model for prior preservation. Default: `None`.
        image_transforms (torch.nn.Module): Torchvision transforms to apply to images.
            Default: `None`.
    """

    def __init__(self,
                 instance_data_root: str,
                 instance_prompt: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 class_prompt: Optional[str] = None,
                 class_data_root: Optional[str] = None,
                 image_transforms: Optional[torch.nn.Module] = None):
        self.image_transforms = image_transforms
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index: int):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == 'RGB':
            instance_image = instance_image.convert('RGB')
        example['instance_images'] = self.image_transforms(
            instance_image)  # type:ignore
        example['instance_prompt_ids'] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images])
            if not class_image.mode == 'RGB':
                class_image = class_image.convert('RGB')
            example['class_images'] = self.image_transforms(
                class_image)  # type: ignore
            example['class_prompt_ids'] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
            ).input_ids
        return example
