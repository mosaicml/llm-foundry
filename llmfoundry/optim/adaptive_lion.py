# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from composer.utils import dist
from torch.optim.optimizer import Optimizer

from llmfoundry.optim.outlier_detection import OutlierDetector

log = logging.getLogger(__name__)

__all__ = [
    'DecoupledAdaLRLion',
    'DecoupledClipLion',
]


class DecoupledAdaLRLion(Optimizer):
    """DecoupledAdaLRLion.

    This class implements a variant of Lion which lowers the layerwise
    learning rate when the layer's moment becomes an outlier. A moment is an
    outlier if it is some multiple `outlier_threshold` times larger than the
    simple windowed moving average (MVA) of moment norms taken from steps T-1000
    to T-500. If an outlier is detected, the LR is lowered by `lr_penalty` for
    `timeout` steps. If N outliers are detected within `timeout` steps, the LR
    is scaled down by max(`lr_penalty` ** N, `min_scale`).

    Args:
        params (Iterable[torch.Parameter]): Model parameters to optimize
        lr (float): Learning rate for updates
        betas (Tuple[float]): Momentum factors
        weight_decay (float): Weight decay
        outlier_threshold (float): Multiplicative factor determining what constitutes an "outlier" relative to the MVA of gradient norms.
        timeout (int): Number of steps to lower the learning for after seeing an outlier.
        lr_penalty (float): Multiplicative scale by which to lower the LR for each outlier.
        min_scale (float): Minimum allowed scaling of the LR .
    """
    metric_functions = {
        'l2_norm/moment':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(optim_state['exp_avg']),
        'l2_norm/param':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(param.data),
        'l2_norm/update':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(step_tensor),
        'l2_norm/grad':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(param.grad),
    }

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        outlier_threshold: float = 10.0,
        timeout: int = 100,
        lr_penalty: float = .707,
        min_scale: float = 1e-4,
    ):
        if lr <= 0.:
            raise Exception(f'Invalid LR: {lr}. LR must be > 0')
        if not all(0. <= beta <= 1. for beta in betas):
            raise Exception(
                f'Invalid beta values: {betas} All betas must be between 0 and 1.',
            )
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. Are you sure you want to do this? '
                +
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!',
            )

        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.outlier_threshold = outlier_threshold
        self.timeout = timeout
        self.lr_penalty = lr_penalty
        self.min_scale = min_scale

    @staticmethod
    def lionw(
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        initial_lr: float,
        wd: float,
        beta1: float,
        beta2: float,
    ) -> None:
        # stepweight decay
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        # update is interpolation between gradient and momentum
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # momentum is interp b/w gradient and itself
        exp_avg.lerp_(grad, 1 - beta2)

    @staticmethod
    def adjust_lr(
        lr: float,
        lr_penalty: float,
        num_times: int,
        min_scale: float,
    ) -> float:
        """Adjusts LR.

        Multiplicatively scales down the LR by lr_penalty for each outlier
        that has occurred in the last `timeout` number of steps, capping the
        scaling to be no smaller than `min_scale`.

        Args:
            lr (float): Base learning rate
            lr_penalty (float): Scaling factor to multiply by for each outlier
            num_times (int): Number of outliers in the last `timeout` steps
            min_scale (float): Minimum scaling to apply to our LR.

        Returns:
            float: Scaled LR
        """
        return lr * max(min_scale, lr_penalty**num_times)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(
                lambda p: p.grad is not None and p.requires_grad,
                group['params'],
            ):

                grad, lr, initial_lr, wd, beta1, beta2, state = p.grad, group[
                    'lr'], group['initial_lr'], group[
                        'weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['moment_tracker'] = OutlierDetector(
                        self.outlier_threshold,
                    )
                    state['outlier_timestamp'] = []
                    state['step'] = 0

                exp_avg = state['exp_avg']

                # determine if the new moment resulting from this grad would be an outlier
                moment_norm = torch.linalg.vector_norm(
                    exp_avg.lerp(grad, 1 - beta2),
                )**2

                if dist.get_world_size() > 1:
                    dist.all_reduce(moment_norm, reduce_operation='SUM')
                moment_norm = math.sqrt(moment_norm)

                if state['moment_tracker'].insert_observation(moment_norm):
                    state['outlier_timestamp'].append(state['step'])

                removed = [
                    ts for ts in state['outlier_timestamp']
                    if state['step'] - ts > self.timeout
                ]

                for ts in removed:
                    state['outlier_timestamp'].remove(ts)

                lr = self.adjust_lr(
                    lr,
                    self.lr_penalty,
                    len(state['outlier_timestamp']),
                    self.min_scale,
                )
                self.lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2)
                state['step'] += 1

        return loss

    def dist_reduce_metrics(self, optimizer_metrics: Dict[str, torch.Tensor]):
        for metric in optimizer_metrics:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = torch.tensor(math.sqrt(reduced))
            elif metric.startswith('cosine'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                A_reduced_norm = optimizer_metrics[f'l2_norm/{A}/{layer}']
                B_reduced_norm = optimizer_metrics[f'l2_norm/{B}/{layer}']
                optimizer_metrics[
                    metric] = reduced / (A_reduced_norm * B_reduced_norm)
            elif metric.startswith('layerwise_lr'):
                continue
            else:
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics: Dict[str, torch.Tensor]):
        """Preprocess metrics to reduce across ranks correctly."""
        # Only L2 norm metric keys are present, can skip sorting at this stage
        for metric in optimizer_metrics:
            # L2 norms need to be squared, before they are reduced via summation
            optimizer_metrics[metric] = optimizer_metrics[metric]**2
        return optimizer_metrics

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict,
    ):
        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, _ = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            layerwise_lr = self.adjust_lr(
                lr,
                self.lr_penalty,
                len(param_optim_state['outlier_timestamp']),
                self.min_scale,
            )

            step_tensor = param_optim_state['exp_avg'].clone().lerp_(
                param.grad,
                1 - beta1,
            ).sign_().mul_(lr)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[
                    metric](param, param_optim_state, step_tensor)

            optimizer_metrics[f'layerwise_lr/{name}'] = torch.tensor(
                layerwise_lr,
            )

        return optimizer_metrics


class DecoupledClipLion(Optimizer):
    """DecoupledClipLION.

    This class implements a variant of Lion which clips layerwise gradients
    that are "outliers". A gradient is an outlier if it is some multiple k times
    larger than the simple windowed moving average (MVA) of gradient norms taken
    from steps T-1000 to T-500. If an outlier is detected, it is clipped.

    to no longer have norm k * MVA.

    Args:
        params (Iterable[torch.Parameter]): Model parameters to optimize
        lr (float): Learning rate for updates
        betas (Tuple[float]): Momentum factors
        weight_decay (float): Weight decay
        outlier_threshold (float): Multiplicative factor determining what constitutes an "outlier" relative to the MVA of gradient norms.
    """
    metric_functions = {
        'l2_norm/moment':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(optim_state['exp_avg']),
        'l2_norm/param':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(param.data),
        'l2_norm/update':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(step_tensor),
        'l2_norm/grad':
            lambda param, optim_state, step_tensor: torch.linalg.
            vector_norm(param.grad),
    }

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        outlier_threshold: float = 5.0,
    ):
        if lr <= 0.:
            raise Exception(f'Invalid LR: {lr}. LR must be > 0')
        if not all(0. <= beta <= 1. for beta in betas):
            raise Exception(
                f'Invalid beta values: {betas} All betas must be between 0 and 1.',
            )
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. Are you sure you want to do this? '
                +
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!',
            )

        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.outlier_threshold = outlier_threshold

    @staticmethod
    def lionw(
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        initial_lr: float,
        wd: float,
        beta1: float,
        beta2: float,
    ) -> None:
        # stepweight decay
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        # update is interpolation between gradient and momentum
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # momentum is interp b/w gradient and itself
        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(
                lambda p: p.grad is not None and p.requires_grad,
                group['params'],
            ):

                grad, lr, initial_lr, wd, beta1, beta2, state = p.grad, group[
                    'lr'], group['initial_lr'], group[
                        'weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['grad_tracker'] = OutlierDetector(
                        self.outlier_threshold,
                    )
                    state['clipped_batches'] = torch.tensor(0.0)

                exp_avg = state['exp_avg']

                # determine if the new moment resulting from this grad would be an outlier
                grad_norm = torch.linalg.vector_norm(grad)**2

                if dist.get_world_size() > 1:
                    dist.all_reduce(grad_norm, reduce_operation='SUM')
                grad_norm = math.sqrt(grad_norm)

                if state['grad_tracker'].insert_observation(grad_norm):
                    state['clipped_batches'] += 1.0
                    clip_norm = state['grad_tracker'].get_slow_mva(
                    ) * self.outlier_threshold
                    grad = grad.div(grad_norm).mul_(clip_norm)

                self.lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2)

        return loss

    def dist_reduce_metrics(self, optimizer_metrics: Dict[str, torch.Tensor]):
        local_keys = list(optimizer_metrics.keys())
        all_gathered_keys = dist.all_gather_object(local_keys)
        all_keys = set()
        for keys in all_gathered_keys:
            all_keys.update(keys)

        # Sort keys to ensure every rank has the same keys order
        # Only L2 norm metric keys are present, can apply regular sort
        all_keys = sorted(all_keys)
        for metric in all_keys:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = torch.tensor(math.sqrt(reduced))
            elif metric.startswith('clipped_batches'):
                continue
            else:
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics: Dict[str, torch.Tensor]):
        """Preprocess metrics to reduce across ranks correctly."""
        # Sort L2 norms first so they are squared before other metrics, which depend on squared values
        metrics = optimizer_metrics.keys()
        metrics = sorted(
            metrics,
            key=lambda metric: 0 if 'l2_norm' in metric else 1,
        )
        for metric in metrics:
            if metric.startswith('l2_norm'):
                # L2 norms need to be squared, before they are reduced via summation
                optimizer_metrics[metric] = optimizer_metrics[metric]**2
            elif metric.startswith('cosine'):
                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                # L2 norm would've been squared in previous branch
                A_rank_subset_norm = math.sqrt(
                    optimizer_metrics[f'l2_norm/{A}/{layer}'],
                )
                B_rank_subset_norm = math.sqrt(
                    optimizer_metrics[f'l2_norm/{B}/{layer}'],
                )

                optimizer_metrics[metric
                                 ] *= A_rank_subset_norm * B_rank_subset_norm

        return optimizer_metrics

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict,
    ):
        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, _ = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            step_tensor = param_optim_state['exp_avg'].clone().lerp_(
                param.grad,
                1 - beta1,
            ).sign_().mul_(lr)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[
                    metric](param, param_optim_state, step_tensor)

            optimizer_metrics[f'clipped_batches/{name}'] = param_optim_state[
                'clipped_batches']

        return optimizer_metrics
