# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

# TODO: this should be removed when examples has a setup.py i.e. installable
sys.path.append('.')

import torch
from omegaconf import OmegaConf

from examples.deeplab.main import main
from examples.deeplab.tests.test_data import SynthADE20KDirectory


@pytest.mark.parametrize('recipe_name', [None, 'mild', 'medium', 'hot'])
def test_trainer(recipe_name):
    if recipe_name == 'hot' and not torch.cuda.is_available():
        pytest.xfail(
            'SAM currently requires running with mixed precision due to a composer bug.'
        )

    with open('yamls/deeplabv3.yaml') as f:
        base_config = OmegaConf.load(f)

    with open('tests/smoketest_config.yaml') as f:
        smoke_config = OmegaConf.load(f)

    config = OmegaConf.merge(base_config, smoke_config)
    config.recipe_name = recipe_name

    with SynthADE20KDirectory() as tmp_datadir:
        print(tmp_datadir)
        config.train_dataset.path = tmp_datadir
        config.eval_dataset.path = tmp_datadir
        # Also save checkpoints in the temporary directory
        config.save_folder = tmp_datadir

        # Train
        trainer1 = main(config)
        model1 = trainer1.state.model.module

        # Check that the checkpoint was saved
        chkpt_path = os.path.join(tmp_datadir, 'ep0-ba1-rank0.pt')
        assert os.path.isfile(chkpt_path)

        # Check that the checkpoint was loaded by comparing model weights
        config.load_path = chkpt_path
        config.is_train = False
        config.seed += 10  # change seed
        trainer2 = main(config)
        model2 = trainer2.state.model.module

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(param1, param2)
