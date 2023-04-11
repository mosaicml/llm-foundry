# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from examples.stable_diffusion_dreambooth.model import \
    build_stable_diffusion_model


@pytest.mark.parametrize(
    'model_name',
    ['CompVis/stable-diffusion-v1-4', 'stabilityai/stable-diffusion-2-1'])
def test_model_builder(model_name: str):
    # test that the StableDiffusion base class outputs the correct size outputs
    # for all popular model versions.
    model = build_stable_diffusion_model(model_name, pretrained=False)
    batch_size = 1
    H = 8
    W = 8
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
