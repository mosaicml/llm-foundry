# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Iterable, Optional

import torch


class NoOp(torch.optim.Optimizer):
    """Optimizer that performs no optimization steps."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
    ):
        """Initialize NoOp optimizer.

        Args:
            params (Iterable[torch.Tensor]): Model parameters for the optimizer.
        """
        # LR schedulers expect param groups to have LR. Unused.
        defaults = {'lr': 0.0}
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, dict[Any, Any]]) -> None:
        super().__setstate__(state)

    def state_dict(self):
        return super().state_dict()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform no-op optimization step where no parameters are updated.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        return loss
