# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Callable

import torch
from composer.trainer import Trainer
from torch.utils.data import DataLoader

from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_optimizer


def test_no_op_does_nothing(
    build_tiny_mpt: Callable[..., ComposerMPTCausalLM],
    tiny_ft_dataloader: DataLoader,
):

    # Build MPT model
    model = build_tiny_mpt(
        loss_fn='torch_crossentropy',
        attn_config={
            'attn_impl': 'torch',
        },
    )

    # Build NoOp optimizer
    no_op_optim = build_optimizer(model, 'no_op', optimizer_config={})

    orig_model = copy.deepcopy(model)

    # build trainer
    trainer = Trainer(
        model=model,
        train_dataloader=tiny_ft_dataloader,
        max_duration=f'2ba',
        optimizers=no_op_optim,
    )
    trainer.fit()

    # Check that the model has not changed
    for (
        (orig_name, orig_param),
        (new_name, new_param),
    ) in zip(orig_model.named_parameters(), model.named_parameters()):
        print(f'Checking {orig_name} and {new_name}')
        assert torch.equal(orig_param, new_param)
