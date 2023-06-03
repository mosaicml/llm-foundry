
import copy
import time

import numpy as np
import pytest
import torch

from llmfoundry.optim import Lion8bit

import warnings
warnings.filterwarnings('ignore')

# these are chosen based on 32x32 being the breakpoint where quantization
# actually kicks in, and 1024 being the CUDA block size
_MANY_PARAM_SHAPES = [(1, 1), (1, 2), (2, 1), (17, 23), (32, 32),
                      (64, 32), (63, 63), (64, 64)]

np.set_printoptions(linewidth=160, formatter={'float': lambda f: f'{f:5.3f}'})

def _print_excerpt(name: str, t: torch.Tensor, numel: int = 16) -> None:
    print(f'{name}:\t{t.detach().cpu().numpy()[:numel]}')


@pytest.mark.parametrize('N,D', _MANY_PARAM_SHAPES)
@pytest.mark.parametrize('fused', [False, True])
def test_modifies_weights_and_momentums(N: int, D: int, fused: bool) -> None:
    device = 'cuda'
    torch.manual_seed(123)
    X = torch.randn((N, D), device=device, requires_grad=False)
    W = torch.randn((D, D), device=device, requires_grad=True)
    W_orig = W.detach().clone()

    opt = Lion8bit([W], fused=fused, betas=(.9, .99))

    Y = X @ W
    loss = Y.sum()
    loss.backward()
    torch.testing.assert_close(W_orig, W) # no weight modification yet
    opt.step()
    opt.zero_grad()

    with pytest.raises(AssertionError): # opt step modified the weights
        torch.testing.assert_close(W_orig, W)

    # every momentum should be nonzero with infinite precision, but
    # might be zero after quantization
    momentum = opt.state[W]['exp_avg'].materialize()
    assert momentum.shape == (D, D)
    momentum = momentum.ravel()
    assert momentum is not None
    if momentum.numel() == 1:
        assert momentum.item() != 0
    else:
        assert torch.std(momentum).item() > 0


@pytest.mark.parametrize('N,D', _MANY_PARAM_SHAPES)
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('l2_penalty', [0, .1])
@pytest.mark.parametrize('weight_decay', [0, .1])
def test_changes_with_zero_grads(N: int, D: int, fused: bool, l2_penalty: float,
                                 weight_decay: float) -> None:
    torch.manual_seed(123)
    W = torch.rand((D, D), device='cuda', requires_grad=True)
    with torch.no_grad():
        W += torch.sign(W)  # bound away from zero so decay won't change sign
    W_orig = W.detach().clone()

    opt = Lion8bit([W], fused=fused, betas=(.5, .5),
                           l2_penalty=l2_penalty, weight_decay=weight_decay)

    zeros_grad = torch.zeros_like(W)
    for i in range(5):
        W.grad = zeros_grad
        opt.step()
        opt.zero_grad()

        # every momentum should be zero if there's no l2 penalty; an l2 penalty
        # is part of the gradient, so it yields nonzero momentum
        mom = opt.state[W]['exp_avg']
        if l2_penalty == 0:
            assert torch.all(mom.materialize() == 0)
            if mom.is_quantized():
                assert torch.all(mom.quantized == 0)
        else:
            assert mom.materialize().std() != 0
            if mom.is_quantized():
                assert mom.quantized.float().std() != 0

        if l2_penalty or weight_decay:
            assert torch.all(W_orig.abs() > W.abs())
        else:
            torch.testing.assert_close(W_orig, W) # no weight modification


@pytest.mark.parametrize('N,D', [(17, 23), (32, 32), (64, 32), (63, 63), (64, 64)])
def test_descends(N: int, D: int) -> None:
    torch.manual_seed(123)
    device = 'cuda'
    X = torch.randn((N, D), device=device, requires_grad=False)
    W = torch.randn((D, D), device=device, requires_grad=True)

    # we use tiny beta1 so we move almost entirely in the gradient direction
    opt = Lion8bit([W], lr=1e-3, betas=(.9, .9))

    prev_loss = np.inf
    prev_momentum = None
    for it in range(10):
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
        momentum = opt.state[W]['exp_avg'].materialize()
        assert momentum is not None and momentum.shape == W.shape
        if prev_momentum is not None:
            momentum_changes = momentum - prev_momentum
            assert torch.all(momentum_changes >= 0)
            assert momentum_changes.max() > 0
            prev_momentum = momentum


@pytest.mark.parametrize('grad_strategy', ['zero', 'ones', 'const', 'rand'])
def test_lion8b_fused_unfused_unquantized_same(grad_strategy: str, N: int = 64, D: int = 64) -> None:
    torch.manual_seed(123)
    device = 'cuda'

    # each optimizer gets a different weight matrix to optimize
    W0 = torch.rand((D, D), device=device, requires_grad=False, dtype=torch.float32)
    W0.add_(W0.sign())  # bound away from zero so decay won't flip sign
    W_uq = torch.empty_like(W0, requires_grad=True)  # unquantized
    W_uf = torch.empty_like(W0, requires_grad=True)  # unfused
    W_fq = torch.empty_like(W0, requires_grad=True)  # fused and quantized
    W_sgd = torch.empty_like(W0, requires_grad=True)
    with torch.no_grad():
        W_uq.copy_(W0)
        W_uf.copy_(W0)
        W_fq.copy_(W0)
        W_sgd.copy_(W0)

    # we use a high LR, low betas, and regularization so that there will
    # hopefully be differences if *any* of the logic is wrong
    lr = .1
    weight_decay = .01
    kwargs = dict(lr=lr, weight_decay=weight_decay, l2_penalty=.01, betas=(.1, .1))
    opt_uq = Lion8bit([W_uq], quantize=False, **kwargs)
    opt_uf = Lion8bit([W_uf], fused=False, **kwargs)
    opt_fq = Lion8bit([W_fq], fused=True, **kwargs)
    opt_sgd = torch.optim.SGD([W_sgd], lr=lr)

    W_list = [W_uq, W_uf, W_fq, W_sgd]
    opt_list = [opt_uq, opt_uf, opt_fq, opt_sgd]

    if grad_strategy == 'zero':
        grads = torch.zeros_like(W0)
    elif grad_strategy == 'ones':
        grads = torch.ones_like(W0)
    elif grad_strategy == 'const':
        # arange makes blocks have different distros, so we can't
        # get away with bugs like always using the first scale_scale
        grads = torch.arange(N * D, device=device, requires_grad=False, dtype=W0.dtype)
        grads = grads.view(N, D)

    for it in range(10):
        if grad_strategy == 'rand':
            grads = torch.rand((N, D), device=device, requires_grad=False, dtype=W0.dtype)
        for W, opt in zip(W_list, opt_list):
            W.grad = grads.clone()
            opt.step()
            opt.zero_grad()

        diffs_true = (W_uq - W0).ravel()
        diffs_uf = (W_uf - W0).ravel()
        diffs_fq = (W_fq - W0).ravel()
        diffs_sgd = (W_sgd - W0).ravel()

        # fused and unfused should be almost identical; the only differences
        # are intermediate upcasting in the fused impl
        assert torch.cosine_similarity(diffs_uf, diffs_fq, dim=-1) > .99
        torch.testing.assert_close(diffs_uf, diffs_fq, atol=1e-6, rtol=1e-5)

        # fused impl should be close to unfused version with no quantization
        # at all; latter is "ground truth"
        assert torch.cosine_similarity(diffs_true, diffs_fq, dim=-1) > .99

    with pytest.raises(AssertionError):
        # if sgd weights aren't different than LION weights, we haven't
        # changed them enough to meaningfully test the LION logic
        assert torch.cosine_similarity(diffs_true, diffs_fq, dim=-1) < .99


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('quantized_state', [False, True])
def test_state_dict_save_load(device: str, quantized_state: bool):
    torch.manual_seed(123)
    params = []
    for shape in _MANY_PARAM_SHAPES:
        p = torch.rand(shape, device=device, dtype=torch.float32, requires_grad=True)
        p.grad = torch.zeros_like(p)
        params.append(p)

    # create optimizer and have it step so that state gets populated
    opt = Lion8bit(params)
    opt.step()
    opt.zero_grad()

    # copy state dict into a new instance
    state_dict = opt.state_dict()
    opt_new = Lion8bit(params)
    opt_new.load_state_dict(state_dict)

    for p in params:
        d_orig = opt.state[p]
        d_new = opt_new.state[p]
        assert list(d_orig.keys()) == list(d_new.keys())
        mom_orig = d_orig['exp_avg']
        mom_new = d_new['exp_avg']
        if quantized_state:
            mom_orig.materialize()
            mom_new.materialize()
            assert torch.all(mom_orig.materialize() == mom_new.materialize())
        else:
            torch.testing.assert_close(mom_orig.materialize(),
                                       mom_new.materialize(),
                                       atol=1./(2 * 127),
                                       rtol=np.inf)


@pytest.mark.parametrize('N,D', [(32, 32), (256, 256), (1024, 1024), (4096, 4096), [16384, 16384]])
def test_fused_faster_than_unfused(N: int, D: int, min_elems_traversed: int = 4e9):
    W = torch.randn((N, D), device='cuda', requires_grad=True)
    W.grad = torch.randn((N, D), device='cuda', requires_grad=False)

    num_iters = int(np.ceil(min_elems_traversed / W.grad.numel()))
    num_iters = min(100, num_iters)  # don't take all day when overhead-bound

    times = {}
    kwargs = dict(l2_penalty=0, weight_decay=.01) # common case
    for fused in [True, False, 'NA']:
        if fused == 'NA':
            opt = Lion8bit([W], quantize=False, **kwargs)
        else:
            opt = Lion8bit([W], fused=fused, **kwargs)
        for i in range(3):
            opt.step()  # warmup iters
        torch.cuda.synchronize()
        t_start = time.time()
        for i in range(num_iters):
            opt.step()
        torch.cuda.synchronize()
        t_end = time.time()
        times[fused] = t_end - t_start

    rel_tol = 1.05  # within 5% is okay
    abs_tol = .05  # empirical; somehow way more overhead for our numba kernel
    assert times[True] < times[False]
    assert times[True] < times['NA'] * rel_tol + abs_tol
