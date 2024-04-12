# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

import torch

from llmfoundry.layers_registry import fcs, norms
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


def build_fc(
    name: str,
    in_features: int,
    out_features: int,
    fc_kwargs: Dict[str, Any],
):
    kwargs = {
        'in_features': in_features,
        'out_features': out_features,
        **fc_kwargs,
    }

    return construct_from_registry(name=name,
                                   registry=fcs,
                                   pre_validation_function=torch.nn.Module,
                                   kwargs=kwargs)
