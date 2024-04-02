# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import torch

from llmfoundry.layers_registry import norms
from llmfoundry.utils.registry_utils import construct_from_registry


def build_norm(
    name: str,
    normalized_shape: Union[int, List[int], torch.Size],
    device: Optional[str] = None,
):
    kwargs = {
        'normalized_shape': normalized_shape,
        'device': device,
    }

    return construct_from_registry(name=name,
                                   registry=norms,
                                   pre_validation_function=torch.nn.Module,
                                   kwargs=kwargs)
