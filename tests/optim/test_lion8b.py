# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import time
import warnings

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (  # type:ignore .api not in public API
    FullOptimStateDictConfig, LocalOptimStateDictConfig,
    ShardedOptimStateDictConfig)

from llmfoundry.optim import DecoupledLionW
from llmfoundry.optim import DecoupledLionW_8bit as Lion8bit

warnings.filterwarnings('ignore')

_MANY_PARAM_SHAPES = [(1, 1), (1, 2), (17, 23), (64, 32)]
_FLOAT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

np.set_printoptions(linewidth=160, formatter={'float': lambda f: f'{f:5.3f}'})


@pytest.mark.gpu
@pytest.mark.parametrize('N,D', _MANY_PARAM_SHAPES)
@pytest.mark.parametrize('dtype', _FLOAT_DTYPES)
@pytest.mark.parametrize('fused,use_errors', [(False, False), (True, False),
                                              (True, True)])
def test_modifies_weights_and_momentums(N: int, D: int, dtype: torch.dtype,
                                        fused: bool, use_errors: bool) -> None:
    device = 'cuda'
    torch.manual_seed(123)
    X = torch.randn((N, D), device=device, requires_grad=False, dtype=dtype)
    W = torch.randn((D, D), device=device, requires_grad=True, dtype=dtype)
    W_orig = W.detach().clone()

    opt = Lion8bit([W],
                   lr=1.0,
                   _fused=fused,
                   betas=(.75, .75),
                   weight_decay=.2,
                   error_correction=use_errors)

    Y = X @ W
    loss = Y.sum()
    loss.backward()
    torch.testing.assert_close(W_orig, W)  # no weight modification yet
    opt.step()
    opt.zero_grad()

    with pytest.raises(AssertionError):  # opt step modified the weights
        torch.testing.assert_close(W_orig, W)

    # Every momentum should be nonzero with infinite precision, but
    # might be zero after quantization. We turn the _MaybeQuantizedTensor
    # instance into a regular torch Tensor to simplify this check.
    param_state = opt.state[W]  # type:ignore using tensor as key
    momentum = param_state['exp_avg'].materialize()
    assert momentum.shape == (D, D)
    momentum = momentum.ravel()
    if momentum.numel() == 1:
        assert momentum.item() != 0
    else:
        assert torch.std(momentum).item() > 0


@pytest.mark.parametrize('N,D', _MANY_PARAM_SHAPES)
@pytest.mark.parametrize('dtype', [
    torch.float32,
    pytest.param(torch.bfloat16, marks=pytest.mark.gpu),
    pytest.param(torch.float16, marks=pytest.mark.gpu),
    pytest.param(torch.float32, marks=pytest.mark.gpu)
])
@pytest.mark.parametrize('weight_decay', [0, .1])
@pytest.mark.parametrize('fused,use_errors', [(False, False), (True, False),
                                              (True, True)])
def test_changes_with_zero_grads(N: int, D: int, dtype: torch.dtype,
                                 weight_decay: float, fused: bool,
                                 use_errors: bool) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if (device == 'cpu') and (fused or use_errors):
        return

    torch.manual_seed(123)
    W = torch.rand((D, D), device=device, requires_grad=True)
    W_orig = W.detach().clone()

    opt = Lion8bit([W],
                   _fused=fused,
                   betas=(.5, .5),
                   quantize=(device != 'cpu'),
                   weight_decay=weight_decay,
                   error_correction=use_errors)

    zeros_grad = torch.zeros_like(W)
    for _ in range(5):
        W.grad = zeros_grad
        opt.step()
        opt.zero_grad()

        mom = opt.state[W]['exp_avg']  # type:ignore using tensor as key
        assert torch.all(mom.materialize() == 0)
        if mom.is_quantized():
            assert torch.all(mom.quantized == 0)

        if weight_decay:
            assert torch.all(W_orig.abs() > W.abs())
        else:
            torch.testing.assert_close(W_orig, W)  # no weight modification


@pytest.mark.parametrize('N,D', [(1, 8), (17, 23), (32, 32)])
@pytest.mark.parametrize('dtype', [
    torch.float32,
    pytest.param(torch.bfloat16, marks=pytest.mark.gpu),
    pytest.param(torch.float16, marks=pytest.mark.gpu),
    pytest.param(torch.float32, marks=pytest.mark.gpu)
])
@pytest.mark.parametrize('fused,use_errors', [(False, False), (True, False),
                                              (True, True)])
def test_descends(N: int, D: int, dtype: torch.dtype, fused: bool,
                  use_errors: bool) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if (device == 'cpu') and (fused or use_errors):
        return
    torch.manual_seed(123)
    X = torch.randn((N, D), device=device, requires_grad=False, dtype=dtype)
    W = torch.randn((D, D), device=device, requires_grad=True, dtype=dtype)

    # we use tiny beta1 so we move almost entirely in the gradient direction
    opt = Lion8bit([W],
                   lr=1e-2,
                   betas=(.5, .5),
                   quantize=(device != 'cpu'),
                   _fused=fused,
                   error_correction=use_errors)

    prev_loss = np.inf
    prev_momentum = None
    num_iters = 10 if device == 'cuda' else 2  # keep test fast
    for _ in range(num_iters):
        Y = X @ W
        loss = (Y * Y).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        loss_val = loss.item()
        assert loss_val < prev_loss
        prev_loss = loss_val

        # since we're getting the same batch every time and have a small
        # learning rate, our gradients should point in the same direction
        # at each step. Consequently, our momentum should grow each step.
        state_for_param = opt.state[W]  # type:ignore using tensor as key
        momentum = state_for_param['exp_avg'].materialize()
        assert momentum is not None and momentum.shape == W.shape
        if prev_momentum is not None:
            momentum_abs_changes = (momentum - prev_momentum).abs()
            assert torch.all(momentum_abs_changes >= 0)
            assert momentum_abs_changes.max() > 0
        prev_momentum = momentum.clone()  # {gpu, f32 on cpu} write in place


def _nmse(vals_true: torch.Tensor,
          vals_hat: torch.Tensor,
          norm_how: str = 'l2_sq'):
    diffs = vals_true - vals_hat
    mse = (diffs * diffs).mean()
    if norm_how == 'var':
        return mse / vals_true.var()
    return mse / (vals_true * vals_true).mean()


@pytest.mark.gpu
@pytest.mark.parametrize('w_init', ['cyclic', 'rand'])
@pytest.mark.parametrize('grad_strategy', ['zero', 'ones', 'const', 'rand'])
@pytest.mark.parametrize('D', [4, 12])  # vectorized and unvectorized impls
@pytest.mark.parametrize('dtype', _FLOAT_DTYPES)
def test_lion8b_fused_unfused_unquantized_same(w_init: str, grad_strategy: str,
                                               D: int,
                                               dtype: torch.dtype) -> None:
    torch.manual_seed(123)
    device = 'cuda'

    # each optimizer gets a different copy of the weight matrix to optimize
    if w_init == 'cyclic':
        W0 = torch.arange(D * D,
                          device=device,
                          requires_grad=False,
                          dtype=dtype).reshape(D, D)
        W0 = ((W0 // 2 % 3) - 1).to(dtype=dtype)
    elif w_init == 'rand':
        W0 = torch.rand(
            size=(D, D), device=device, requires_grad=False,
            dtype=dtype) * 2 - 1
        W0 += .01 * torch.sign(W0)  # bound away from 0 to cap rel errors
        W0 = W0.to(dtype=dtype)
    else:  # here for pyright
        raise ValueError('Unrecognized w_init: ', w_init)
    W0.add_(W0.sign())  # bound away from zero so decay won't flip sign
    W_true = torch.empty_like(W0, requires_grad=True,
                              dtype=torch.float32)  # ground truth
    W_uq = torch.empty_like(W0, requires_grad=True)  # unquantized
    W_uf = torch.empty_like(W0, requires_grad=True)  # unfused
    W_fq = torch.empty_like(W0, requires_grad=True)  # fused and quantized
    W_fqe = torch.empty_like(W0, requires_grad=True)  # fused, quantized, ecc
    W_sgd = torch.empty_like(W0, requires_grad=True)
    with torch.no_grad():
        W_true.copy_(W0.to(W_true.dtype))
        W_uq.copy_(W0)
        W_uf.copy_(W0)
        W_fq.copy_(W0)
        W_fqe.copy_(W0)
        W_sgd.copy_(W0)

    # we use a high LR, low betas, and regularization so that there will
    # hopefully be differences if *any* of the logic is wrong
    lr = .1
    weight_decay = .01
    kwargs = {'lr': lr, 'weight_decay': weight_decay, 'betas': (.5, .75)}
    opt_true = Lion8bit([W_true], quantize=False, **kwargs)
    opt_uq = Lion8bit([W_uq], quantize=False, **kwargs)
    opt_uf = Lion8bit([W_uf], _fused=False, **kwargs)
    opt_fq = Lion8bit([W_fq], _fused=True, **kwargs)
    opt_fqe = Lion8bit([W_fqe], _fused=True, error_correction=True, **kwargs)
    opt_sgd = torch.optim.SGD([W_sgd], lr=lr)

    W_list = [W_true, W_uq, W_uf, W_fq, W_fqe, W_sgd]
    opt_list = [opt_true, opt_uq, opt_uf, opt_fq, opt_fqe, opt_sgd]

    if grad_strategy == 'zero':
        grads = torch.zeros_like(W0)
    elif grad_strategy == 'ones':
        grads = ((torch.arange(W0.numel()) % 2) * 2 - 1).reshape(W0.shape)
    elif grad_strategy == 'const':
        # arange makes blocks have different distros, so we can't
        # get away with bugs like always using the first scale_scale
        grads = torch.arange(W0.numel(),
                             device=device,
                             requires_grad=False,
                             dtype=W0.dtype).view(W0.shape)
    # next two conditions are just here for pyright
    elif grad_strategy == 'rand':
        grads = torch.tensor([-1])
    else:
        raise ValueError('bad grad_strategy: ', grad_strategy)

    for _ in range(4):
        if grad_strategy == 'rand':  # type:ignore (reportUnnecessaryComparison)
            grads = torch.rand(W0.shape,
                               device=device,
                               requires_grad=False,
                               dtype=W0.dtype)
        for W, opt in zip(W_list, opt_list):
            W.grad = grads.clone().to(dtype=W.dtype, device=W.device)
            opt.step()
            opt.zero_grad()

        W0_f = W0.float()
        diffs_true = (W_true.detach().float() - W0_f).ravel()
        diffs_uq = (W_uq.detach().float() - W0_f).ravel()
        diffs_uf = (W_uf.detach().float() - W0_f).ravel()
        diffs_fq = (W_fq.detach().float() - W0_f).ravel()
        diffs_fqe = (W_fqe.detach().float() - W0_f).ravel()
        diffs_sgd = (W_sgd.detach().float() - W0_f).ravel()

        # a bunch of made-up numbers; should be tight enough to detect
        # regressions, but aren't enough to 100% guarantee correct numerics
        if dtype != torch.bfloat16:
            min_cossim = .99
            max_nmse = .01
        else:
            min_cossim = .98
            max_nmse = .04

        cossim = torch.cosine_similarity  # avoid ugly linewraps

        assert cossim(diffs_true, diffs_uq, dim=-1) > min_cossim
        assert _nmse(diffs_true, diffs_uq) < max_nmse

        assert cossim(diffs_true, diffs_uf, dim=-1) > min_cossim
        assert _nmse(diffs_true, diffs_uf) < max_nmse

        # fused and unfused should be almost identical; the only differences
        # are intermediate upcasting in the fused impl
        assert cossim(diffs_uf, diffs_fq, dim=-1) > min_cossim
        assert _nmse(diffs_uf, diffs_fq) < max_nmse

        # fused impl should be close to unfused version with no quantization
        # at all; latter is "ground truth"
        assert cossim(diffs_true, diffs_fq, dim=-1) > min_cossim
        assert _nmse(diffs_true, diffs_fq) < max_nmse

        # fused impl with errors should also be close to "true" updates;
        assert cossim(diffs_true, diffs_fqe, dim=-1) > min_cossim
        assert _nmse(diffs_true, diffs_fqe) < max_nmse

        # error correction should reduce error, or at least do no worse
        assert _nmse(diffs_true, diffs_fqe) <= _nmse(diffs_true, diffs_fq)

    # if sgd weights aren't different than LION weights, we haven't
    # changed them enough to meaningfully test the LION logic
    if grad_strategy not in ('zero', 'ones'):
        assert torch.cosine_similarity(
            diffs_true,  # type:ignore (reportUnboundVariable)
            diffs_sgd,  # type:ignore (reportUnboundVariable)
            dim=-1) < .99
        assert _nmse(
            diffs_true,  # type:ignore (reportUnboundVariable)
            diffs_sgd  # type:ignore (reportUnboundVariable)
        ) > .01


@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('quantized_state', [False, True])
@pytest.mark.parametrize('dtype', _FLOAT_DTYPES)
@pytest.mark.parametrize('use_errors', [False, True])
def test_state_dict_save_load(device: str, quantized_state: bool,
                              dtype: torch.dtype, use_errors: bool):
    torch.manual_seed(123)
    params = []
    for shape in _MANY_PARAM_SHAPES:
        p = torch.rand(shape, device=device, dtype=dtype, requires_grad=True)
        p.grad = torch.rand_like(p)
        params.append(p)

    # create optimizer and have it step so that state gets populated
    opt = Lion8bit(params,
                   compress_state_dict=quantized_state,
                   error_correction=use_errors)
    if device == 'cpu':
        with pytest.raises(NotImplementedError):
            opt.step()
        return
    else:
        opt.step()
    opt.zero_grad()

    # copy state dict into a new instance
    state_dict = opt.state_dict()
    opt_new = Lion8bit(params,
                       compress_state_dict=quantized_state,
                       error_correction=use_errors)
    opt_new.load_state_dict(state_dict)

    for p in params:
        d_orig = opt.state[p]
        d_new = opt_new.state[p]
        assert list(d_orig.keys()) == list(d_new.keys())
        mom_orig = d_orig['exp_avg']
        mom_new = d_new['exp_avg']
        if quantized_state:
            # Optimizer load_state_dict insists on converting scales to
            # dtype of param, which is lossy for bf16 params.
            # Ideally we'd require == for everything but it's less complexity
            # to just relax the bf16 test
            assert torch.all(mom_orig.quantized == mom_new.quantized)
            if dtype == torch.bfloat16:
                torch.testing.assert_close(mom_orig.scales,
                                           mom_new.scales,
                                           atol=1e-3,
                                           rtol=1e-2)
            else:
                assert torch.all(mom_orig.scales == mom_new.scales)

        torch.testing.assert_close(mom_orig.materialize(),
                                   mom_new.materialize(),
                                   atol=1. / (2 * 127),
                                   rtol=np.inf)
        if use_errors and (dtype != torch.float32):
            torch.testing.assert_close(d_orig['errors'], d_new['errors'])


class _DummyModule(nn.Module):

    def __init__(self, device: str, dtype: torch.dtype):
        super().__init__()
        self.linear0 = nn.Linear(4, 3, device=device, dtype=dtype)
        self.linear1 = nn.Linear(3, 5, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type:ignore
        return self.linear1(self.linear0(x))


_FULL_STATE = fsdp.StateDictType.FULL_STATE_DICT
_SHARDED_STATE = fsdp.StateDictType.SHARDED_STATE_DICT
_LOCAL_STATE = fsdp.StateDictType.LOCAL_STATE_DICT


@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize('dtype', _FLOAT_DTYPES)
@pytest.mark.parametrize('use_errors', [False, True])
@pytest.mark.parametrize('state_sharding',
                         [_FULL_STATE, _SHARDED_STATE, _LOCAL_STATE])
@pytest.mark.parametrize('save_as_lion8b, load_as_lion8b', [(False, True),
                                                            (True, False),
                                                            (True, True)])
def test_fsdp_save_load(dtype: torch.dtype, use_errors: bool,
                        state_sharding: fsdp.StateDictType,
                        save_as_lion8b: bool, load_as_lion8b: bool):
    device = 'cuda'
    if torch.cuda.device_count() < 2:
        pytest.skip(f'This test requires 2+ GPUs.')

    torch.cuda.set_device(f'cuda:{os.environ["RANK"]}')  # needed for fsdp
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    assert dist.get_world_size() >= 2, 'Misconfigured test run!'

    # nb: this is the line that causes:
    #   `Warning: Deallocating Tensor that still has live PyObject references.`
    # suggesting this warning isn't an issue with our test code. It's also
    # going to stdout (probably from cpp) so we can't suppress it with warnings
    mod = FSDP(_DummyModule(device=device, dtype=dtype))

    # actual forward pass instead of setting p.grad to avoid FSDP issues
    X = torch.rand(size=(5, 4), device=device, dtype=dtype)
    Y = mod(X)
    Y.sum().backward()
    for p in mod.parameters():
        p.grad = torch.rand_like(p)

    # create optimizer and have it step so that state gets populated
    if save_as_lion8b:
        opt = Lion8bit(mod.parameters(), error_correction=use_errors)
    else:
        opt = DecoupledLionW(mod.parameters())
    opt.step()
    opt.zero_grad()

    def _set_state_dict_type(model: nn.Module):
        # for mapping between state dict types and optim state dict types, see:
        # https://github.com/pytorch/pytorch/blob/a815e719e85899d4229616617e7827d4de191c2d/torch/distributed/fsdp/fully_sharded_data_parallel.py#L664 # noqa
        state_dict_cfg = {
            _FULL_STATE: fsdp.FullStateDictConfig(rank0_only=False),
            _SHARDED_STATE: fsdp.ShardedStateDictConfig(),
            _LOCAL_STATE: fsdp.LocalStateDictConfig(),
        }[state_sharding]
        optim_cfg = {
            _FULL_STATE: FullOptimStateDictConfig(rank0_only=False),
            _SHARDED_STATE: ShardedOptimStateDictConfig(),
            _LOCAL_STATE: LocalOptimStateDictConfig(),
        }[state_sharding]
        FSDP.set_state_dict_type(model, state_sharding, state_dict_cfg,
                                 optim_cfg)

    def _local_shard(t: torch.Tensor) -> torch.Tensor:
        try:  # can't operate on ShardedTensors directly
            return t.local_tensor()  # type: ignore
        except AttributeError:
            return t

    # load FSDP state dict
    _set_state_dict_type(mod)
    opt_state_dict = FSDP.optim_state_dict(mod, opt)

    # make a new model and optimizer
    mod_new = FSDP(_DummyModule(device=device, dtype=dtype))
    if load_as_lion8b:
        opt_new = Lion8bit(mod_new.parameters(), error_correction=use_errors)
    else:
        opt_new = DecoupledLionW(mod_new.parameters())
    _set_state_dict_type(mod_new)

    # load state dict into the new optimizer
    opt_state_dict_slice = FSDP.optim_state_dict_to_load(
        optim_state_dict=opt_state_dict, model=mod_new, optim=opt_new)
    opt_new.load_state_dict(opt_state_dict_slice)

    new_opt_state_dict = FSDP.optim_state_dict(mod_new, opt_new)

    orig_state = opt_state_dict['state']
    orig_param_groups = opt_state_dict['param_groups']
    new_state = new_opt_state_dict['state']
    new_param_groups = new_opt_state_dict['param_groups']

    all_keys = set(orig_state.keys()) | set(new_state.keys())
    assert orig_param_groups == new_param_groups  # works since strs, not ptrs
    for k in all_keys:  # keys are param paths in module as strings
        d_orig = orig_state[k]
        d_new = new_state[k]
        assert list(d_orig.keys()) == list(d_new.keys())
        mom_orig = d_orig['exp_avg']
        mom_new = d_new['exp_avg']

        assert mom_orig.shape == mom_new.shape
        both_lion8b = save_as_lion8b and load_as_lion8b
        check_errors = both_lion8b and use_errors and (dtype != torch.float32)
        if both_lion8b:
            assert mom_orig.dtype == mom_new.dtype
            if check_errors:
                errs_orig = d_orig['errors']
                errs_new = d_new['errors']
                assert errs_orig.shape == errs_new.shape
                assert errs_orig.dtype == errs_new.dtype

        # momentums may not be bit-for-bit identical because Optimizer upcasts
        # to f32 and we convert back to bf16, possibly with different rounding
        torch.testing.assert_close(_local_shard(mom_orig).float(),
                                   _local_shard(mom_new).float(),
                                   atol=1e-4,
                                   rtol=1. / 128)
        # errors not bit-for-bit identical because scales get upcast too
        if check_errors:
            torch.testing.assert_close(_local_shard(d_orig['errors']),
                                       _local_shard(d_new['errors']))


@pytest.mark.gpu
@pytest.mark.parametrize('N,D', [(32, 32), (256, 256), (1024, 1024),
                                 (4096, 4096), [16384, 16384]])
def test_fused_as_fast_as_unfused(N: int,
                                  D: int,
                                  min_elems_traversed: int = 1000000):

    def _time_kernels(N: int, D: int, min_elems_traversed: int):
        W = torch.randn((N, D), device='cuda', requires_grad=True)
        W.grad = torch.randn((N, D), device='cuda', requires_grad=False)

        num_iters = int(np.ceil(min_elems_traversed / W.grad.numel()))
        num_iters = min(100,
                        num_iters)  # don't take all day when overhead-bound

        times = {}
        kwargs = {'weight_decay': .01}
        combos = [(True, False), (True, True), (False, False), ('NA', False)]
        for fused, use_errors in combos:
            if fused == 'NA':
                opt = Lion8bit(
                    [W], quantize=False,
                    **kwargs)  # type:ignore (reportGeneralTypeIssues)
            else:
                opt = Lion8bit(
                    [W], _fused=fused, error_correction=use_errors,
                    **kwargs)  # type:ignore (reportGeneralTypeIssues)
            for _ in range(3):
                opt.step()  # warmup iters
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(num_iters):
                opt.step()
            torch.cuda.synchronize()
            t_end = time.time()
            dur = (t_end - t_start) / num_iters
            if use_errors:
                times['ecc'] = dur
            else:
                times[fused] = dur
        return times

    times = _time_kernels(N, D, min_elems_traversed)

    atol = 2e-4  # should always be faster, but atol helps avoid flakiness
    it = 0
    while True:
        try:
            assert times[True] < times[False] + atol
            assert times[True] < times['NA'] + atol
            assert times['ecc'] < times['NA'] + atol
            print('')
            print('time fused (ms):       ', times[True] * 1e3)
            print('time fused+ecc (ms):   ', times['ecc'] * 1e3)
            print('time unfused (ms):     ', times[False] * 1e3)
            print('time unquantized (ms): ', times['NA'] * 1e3)
            break
        except AssertionError as e:
            if it >= 2:  # allow 3 retries to avoid flakiness
                raise e
        times = _time_kernels(N, D, min_elems_traversed)
        it += 1
