# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

__all__ = [
    'dMoE',
    'LearnedRouter',
    'MLP',
    'GLU',
    'DroplessMLP',
]

DEFAULT_ACTIVATION_FN = partial(F.gelu, approximate='tanh')


# Add option to route tokens uniformly across experts. We use
# a custom autograd op router backwards is still run for benchmarking.
class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,  # pyright: ignore[reportMissingParameterType]
        x: torch.Tensor,
        num_experts: int,
    ):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


class LearnedRouter(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_jitter_eps: Optional[float],
        moe_normalize_expert_weights: Optional[Union[int, float]],
        uniform_expert_assignment: bool,
        device: Optional[torch.device],
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.moe_num_experts: int = moe_num_experts
        self.moe_top_k: int = moe_top_k
        self.moe_jitter_eps: Optional[float] = moe_jitter_eps
        self.moe_normalize_expert_weights: Optional[Union[
            int, float]] = moe_normalize_expert_weights
        self.uniform_expert_assignment: bool = uniform_expert_assignment

        self.layer: torch.nn.Module = torch.nn.Linear(
            hidden_size,
            moe_num_experts,
            bias=False,
            device=device,
        )

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        assert self.moe_jitter_eps is not None
        low: float = 1.0 - self.moe_jitter_eps
        high: float = 1.0 + self.moe_jitter_eps
        noise: torch.Tensor = torch.rand(
            x.size(),
            dtype=x.dtype,
            device=x.device,
        )
        return low + noise * (high - low)

    def _top_k(self, scores: torch.Tensor) -> torch.Tensor:
        if self.moe_top_k == 1:
            return scores.max(
                dim=-1,
            )  # pyright: ignore[reportGeneralTypeIssues]
        return torch.topk(
            scores,
            self.moe_top_k,
            dim=-1,
        )  # pyright: ignore[reportGeneralTypeIssues]

    def forward(self, x: torch.Tensor):
        if self.training and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
        expert_weights, top_experts = self._top_k(scores)
        if self.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        top_experts = (
            _UniformExpertAssignment.apply(top_experts, self.moe_num_experts)
            if self.uniform_expert_assignment else top_experts
        )
        scores = scores.to(x.dtype)
        expert_weights = expert_weights.to(x.dtype)
        return scores, expert_weights, top_experts


class MLP(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        moe_num_experts: int,
        activation_fn: Callable,
        device: Optional[torch.device],
    ) -> None:
        super().__init__()

        self.moe_num_experts: int = moe_num_experts
        self.ffn_hidden_size: int = ffn_hidden_size
        self.hidden_size: int = hidden_size
        self.activation_fn: Callable = activation_fn

        self.w1 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.w2 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1 = self.w1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_w2 = self.w2.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]

        before_activation = x @ expert_w1.t()
        layer_1_output = self.activation_fn(before_activation)
        output = layer_1_output @ expert_w2
        return output


class GLU(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        moe_num_experts: int,
        activation_fn: Callable,
        device: Optional[torch.device],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts

        self.w1 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.v1 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.w2 = torch.nn.Parameter(
            torch.rand(
                moe_num_experts * ffn_hidden_size,
                hidden_size,
                device=device,
            ),
        )
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor):
        expert_w1 = self.w1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_v1 = self.v1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_w2 = self.w2.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]

        x1 = x.matmul(expert_w1.t())
        x2 = x.matmul(expert_v1.t())
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = x1.matmul(expert_w2)
        return x1


class DroplessMLP(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        mlp_type: str,
        moe_num_experts: int,
        activation_fn: Callable,
        bias: bool,
        device: Optional[torch.device],
    ):
        super().__init__()
        self.moe_num_experts = moe_num_experts

        if mlp_type == 'mlp':
            self.mlp = MLP(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                moe_num_experts=moe_num_experts,
                activation_fn=activation_fn,
                device=device,
            )
        elif mlp_type == 'glu':
            self.mlp = GLU(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                moe_num_experts=moe_num_experts,
                activation_fn=activation_fn,
                device=device,
            )
        else:
            raise ValueError(f'Received unknown {mlp_type=}')

    def forward(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        expert_weights: torch.Tensor,
        top_experts: torch.Tensor,
    ):
        in_shape = x.shape
        hidden_size = in_shape[-1]

        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = torch.nn.functional.one_hot(
            top_experts,
            num_classes=self.moe_num_experts,
        )
        
        if expert_mask.dim() == 3:
            expert_mask = expert_mask.permute(2, 1, 0)
        elif expert_mask.dim() == 2:
            expert_mask = expert_mask.t()
        else:
            raise ValueError(f'Unexpected expert mask dimensions of {expert_mask.dims()}')
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue
            # In torch it is faster to index using lists than torch tensors
            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            mlp_output = self.mlp(expert_tokens, expert_idx)
            expert_out = mlp_output * expert_weights[token_list, topk_list,
                                                     None]

            out.index_add_(0, token_idx, expert_out)

        out = out.view(in_shape)
        return out


class dMoE(torch.nn.Module):

    def __init__(
        self,
        device: Optional[torch.device],
        hidden_size: int = 1024,
        ffn_hidden_size: int = 4096,
        moe_num_experts: int = 1,
        moe_top_k: int = 1,
        mlp_type: str = 'mlp',
        activation_fn: Callable = DEFAULT_ACTIVATION_FN,
        moe_jitter_eps: Optional[float] = None,
        moe_normalize_expert_weights: Optional[Union[int, float]] = None,
        uniform_expert_assignment: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        # Token router.
        self.router = LearnedRouter(
            hidden_size,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_jitter_eps=moe_jitter_eps,
            moe_normalize_expert_weights=moe_normalize_expert_weights,
            uniform_expert_assignment=uniform_expert_assignment,
            device=device,
        )

        # Expert computation helper.
        self.experts = DroplessMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            mlp_type=mlp_type,
            moe_num_experts=moe_num_experts,
            activation_fn=activation_fn,
            bias=bias,
            device=device,
        )

    def forward(self, x: torch.Tensor):
        # Compute the expert scores and assignments.
        scores, expert_weights, top_experts = self.router(x)
        # Compute the experts.
        return self.experts(x, scores, expert_weights, top_experts)
