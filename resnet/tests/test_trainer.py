# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

# TODO: this should be removed when examples has a setup.py i.e. installable
sys.path.append('.')

import torch
from omegaconf import OmegaConf

from ..main import main
from ..tests.utils import SynthClassificationDirectory


@pytest.mark.parametrize('recipe_name', [None, 'mild', 'medium', 'hot'])
def test_trainer(recipe_name):
    with open('yamls/resnet50.yaml') as f:
        base_config = OmegaConf.load(f)

    with open('tests/smoketest_config.yaml') as f:
        smoke_config = OmegaConf.load(f)
    config = OmegaConf.merge(base_config, smoke_config)
    config.recipe_name = recipe_name

    with SynthClassificationDirectory() as tmp_datadir:
        print(tmp_datadir)
        config.train_dataset.path = tmp_datadir
        config.eval_dataset.path = tmp_datadir
        # Also save checkpoints in the temporary directory
        config.save_folder = tmp_datadir

        # Train
        trainer1 = main(config)
        model1 = trainer1.state.model.module

        # TODO avoid tests taking a long time to exit without this
        trainer1.state.dataloader._iterator._shutdown_workers()  # type: ignore

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
