# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Image captioning dataset creation tools and preprocessing."""

import random

import numpy as np
import torch
import transformers
from composer.core import DataSpec
from composer.utils import dist
from datasets.load import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def collate_fn(examples: dict):
    image_tensor = torch.stack(
        [example['image_tensor'] for example in examples])
    image_tensor = image_tensor.to(
        memory_format=torch.contiguous_format).float()  # type: ignore
    input_ids = torch.stack([example['input_ids'] for example in examples])
    return {'image_tensor': image_tensor, 'input_ids': input_ids}


def build_hf_image_caption_datapsec(name: str,
                                    resolution: int,
                                    tokenizer: transformers.PreTrainedTokenizer,
                                    mean: list = [0.5],
                                    std: list = [0.5],
                                    image_column: str = 'image',
                                    caption_column: str = 'text',
                                    center_crop: bool = True,
                                    random_flip: bool = True,
                                    *,
                                    batch_size: int,
                                    drop_last: bool = True,
                                    shuffle: bool = True,
                                    **dataloader_kwargs):
    """Builds a HuggingFace Image-Caption dataloader.

    Includes transformations and tokenization for diffusion training.

    Args:
        name (str): HuggingFace dataset name or path. Existing datasets can be
            found on the `HuggingFace dataset hub
            <https://huggingface.co/datasets?task_categories=task_categories:text-to-image>`_.
        resolution (int): Final image size after processing.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for the `text_encoder`
            of the model you are training. For a `CLIPTextModel`  this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        mean (list[float]): Dataset channel means for normalization. Default: `[0.5]`.
        std (list[float]): Dataset channel standard deviations for normalization. Default: `[0.5]`.
        image_column (str): Name of the image column in the raw dataset. Default: `image`.
        caption_column (str): Name of the caption column in the raw dataset. Default: `text`.
        center_crop (bool): Whether to center crop the images during preprocessing.
            If `False`, `RandomCrop` will be used. Default: `True`.
        random_flip (bool): Whether to random flip images during processing. Default: `True`.
        batch_size (int): Batch size per device.
        drop_last (bool): Whether to drop last samples. Default: ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the
            dataloader (e.g. num_workers, etc.)
    """
    train_transforms = transforms.Compose([
        transforms.Resize(resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution)
        if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip()
        if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    def tokenize_captions(
            examples: dict,
            tokenizer: transformers.PreTrainedTokenizer = tokenizer,
            caption_column: str = caption_column,
            is_train: bool = True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f'Caption column `{caption_column}` should contain either strings or lists of strings.'
                )
        inputs = tokenizer(captions,
                           max_length=tokenizer.model_max_length,
                           padding='max_length',
                           truncation=True,
                           return_tensors='pt')
        return inputs.input_ids

    def preprocess(examples: dict):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['image_tensor'] = [train_transforms(image) for image in images]
        examples['input_ids'] = tokenize_captions(examples)
        return examples

    if dist.get_world_size() > 1:
        with dist.run_local_rank_zero_first():
            dataset = load_dataset(name, split='train')
    else:
        dataset = load_dataset(name, split='train')

    # add image_tensor and input_ids columns (processed images and text)
    dataset = dataset.with_transform(preprocess)  # type: ignore
    sampler = dist.get_sampler(
        dataset,  # type: ignore
        drop_last=drop_last,  # type: ignore
        shuffle=shuffle)  # type: ignore
    return DataSpec(dataloader=DataLoader(
        dataset=dataset,  # type: ignore
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=collate_fn,  # type: ignore
        **dataloader_kwargs))  # type: ignore


def build_prompt_dataspec(prompts: list[str], batch_size: int,
                          **dataloader_kwargs):
    """Builds a prompt dataset from a list of strings for eval.

    Args:
        prompts (list[str]): A list of prompts.
        batch_size (int): Batch size per device.
    """
    dataset = PromptDataset(prompts)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)
    ds = DataSpec(dataloader=DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        sampler=sampler,
                                        drop_last=False,
                                        **dataloader_kwargs),
                  get_num_samples_in_batch=lambda x: len(x)
                 )  # composer will handle strings in the future.
    return ds


class PromptDataset(Dataset):
    """A simple Dataloader that iterates through a list of strings.

    Args:
        prompts (list[str]): A list of prompts.
    """

    def __init__(self, prompts: list):
        self.prompts = prompts

    def __getitem__(self, index: int):
        return self.prompts[index]

    def __len__(self):
        return len(self.prompts)
