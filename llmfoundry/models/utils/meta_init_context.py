# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
# Modified from https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """Meta initialization context manager.

    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(torch.device('meta'),
                        include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """Device initialization context manager.

    A context manager under which models are initialized with all parameters
    on the specified device.

    Args:
        device (`torch.device`): Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(self: torch.nn.Module, name: str,
                                 param: Optional[torch.nn.Parameter]):
        old_register_parameter(self, name, param)
        if param is not None:
            parameter = self._parameters[name]
            assert parameter is not None

            param_cls = type(parameter)
            kwargs = parameter.__dict__

            self._parameters[name] = param_cls(parameter.to(device), **kwargs)

    def register_empty_buffer(self: torch.nn.Module,
                              name: str,
                              tensor: Optional[torch.Tensor],
                              persistent: bool = True):
        old_register_buffer(self, name, tensor, persistent=persistent)
        if tensor is not None:
            named_buffer = self._buffers[name]
            assert named_buffer is not None
            self._buffers[name] = named_buffer.to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ['empty', 'zeros', 'ones', 'full']
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn: Callable):

        def wrapper(*args: Any, **kwargs: Any):
            kwargs['device'] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch, torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items(
        ):
            setattr(torch, torch_function_name, old_torch_function)
