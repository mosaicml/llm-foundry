# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt and image visualization callback for diffusion models."""
import hashlib
from pathlib import Path

import torchvision.transforms.functional as F
from composer import Callback, Logger, State
from composer.loggers import WandBLogger
from composer.utils import ensure_tuple
from torchvision.utils import make_grid


class LogDiffusionImages(Callback):
    """Logger for diffusion images.

    Logs eval prompts and generated images to a Weights and Biases table at
    the end of an evaluation batch.

    Requires Weights and Biases to be installed and setup.
    """

    def eval_batch_end(self, state: State, logger: Logger):
        prompts = state.batch['prompt']  # batch_size
        # Tensor of shape [len(prompts) * num_images_per_prompt, 3, 512, 512])
        outputs = state.outputs.cpu()  # type: ignore

        num_images_per_prompt = len(outputs) // len(prompts)
        for destination in ensure_tuple(logger.destinations):
            if isinstance(destination, WandBLogger):
                if num_images_per_prompt > 1:
                    outputs = [
                        make_grid(out, nrow=num_images_per_prompt)
                        for out in outputs.chunk(len(prompts))  # type: ignore
                    ]

                for prompt, output in zip(prompts, outputs):
                    destination.log_images(images=output,
                                           name=prompt,
                                           step=state.timestamp.batch.value)


class SaveClassImages(Callback):
    """Logger for saving images on eval batch end.

    Saves images to specified directory, created to build a dataset
    of generated images for prior preservation in Dreambooth training.

    Args:
        class_data_root (str): Directory to save images to.
    """

    def __init__(self, class_data_root: str):
        self.class_data_root = class_data_root
        self.class_images_dir = Path(class_data_root)
        # check current class images incase there are already images in dir
        self.cur_class_images = len(list(self.class_images_dir.iterdir()))

    def eval_batch_end(self, state: State, logger: Logger):
        # Tensor of shape [len(prompts) * num_images_per_prompt, 3, 512, 512])
        images = state.outputs.cpu()  # type: ignore
        batch = state.batch
        for i, image in enumerate(images):
            image = F.to_pil_image(image)
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = self.class_images_dir / f"{batch['index'][i] + self.cur_class_images}-{hash_image}.jpg"
            image.save(image_filename)
