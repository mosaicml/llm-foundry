# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from examples.bert.main import main
from examples.bert.tests.utils import SynthTextDirectory


@pytest.mark.parametrize('model_name,seed', [('mosaic_bert', 17),
                                             ('hf_bert', 18)])
def test_trainer(model_name: str, seed: int):
    with open('tests/smoketest_config_main.yaml') as f:
        config = OmegaConf.load(f)
    assert isinstance(config, DictConfig)
    config.model.name = model_name
    config.seed = seed

    with SynthTextDirectory() as tmp_datadir:
        config.train_loader.dataset.remote = tmp_datadir
        config.train_loader.dataset.local = tmp_datadir
        config.eval_loader.dataset.remote = tmp_datadir
        config.eval_loader.dataset.local = tmp_datadir
        # Also save checkpoints in the temporary directory
        config.save_folder = tmp_datadir

        # Train
        trainer1 = main(config, return_trainer=True)
        assert trainer1 is not None
        model1 = trainer1.state.model.model

        # Check that the checkpoint was saved
        chkpt_path = os.path.join(tmp_datadir, 'latest-rank0.pt')
        assert os.path.isfile(chkpt_path), f'{os.listdir(tmp_datadir)}'

        # Check that the checkpoint was loaded by comparing model weights (with no weight changes)
        config.load_path = chkpt_path
        config.seed += 10  # change seed
        trainer2 = main(config, return_trainer=True, do_train=False)
        assert trainer2 is not None
        model2 = trainer2.state.model.model

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(param1, param2)
