from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch

_KEY_MOMENTUM = 'exp_avg'
_KEY_ERRORS = 'errors'


class DecoupledLionW_8bit(torch.optim.Optimizer):
    """LION optimizer with ~8 bits of state per parameter.

    This optimizer is a drop-in replacement for our regular LION optimizer
    with decoupled weight decay, but uses less memory, writes smaller
    checkpoints, and offers almost-numerically-identical convergence.

    Its state saved per parameter is just an int8, though there are auxiliary
    scaling factors that bring the total memory per parameter to ~8.5 bits.
    The exact quantization scheme is considered an implementation detail
    and may change.

    When training on CPUs, however, no quantization will actually take place.

    See the LION paper (https://arxiv.org/abs/2302.06675) for details about
    the algorithm itself.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (Default: 1e-3)
        betas: two coefficients between 0 and 1 used to combine the current
            gradients and the momentum. The first coefficient is the weight
            of the gradient when computing the update. The second is the
            weight of the gradient when computing the new momentum.
            (Default: .9, .99)
        weight decay: Weights are multiplied by 1 - `weight_decay` after
            each optimizer step. Note that we use decoupled weight decay,
            meaning that this decay does not contribute to the momentum.
            (Default: 0.)
        l2_penalty: adds `l2_penalty * param` to the gradient at the
            start of the optimizer step. This term *is* added to the momentum.
        compress_state_dict: if True, this optimizer's `state_dict` will
            include quantized optimizer states. Otherwise, the optimizer
            states are converted to bfloat16 Tensors matching the shapes of
            their corresponding parameters. The former uses ~8.5 bits per
            parameter while the latter uses 16 bits per parameter. However,
            the former is less thoroughly tested and will not work with
            FSDP or other weight sharding approaches.
        quantize: If False, optimizer states will not actually be quantized.
            This option is available so that one can easily debug whether
            the quantization is causing any convergence issues. Quantization
            is always disabled when training without a CUDA device.
        error_correction: If True, float16 and bfloat16 parameters will be
            given an extra state variable, "errors." This tensor will be
            of the same shape as the parameter but of dtype uint8. This
            auxiliary variable is used to better approximate float32 updates
            by retaining information across optimizer steps.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        compress_state_dict: bool = False,
        quantize: bool = True,
        _fused: bool = True,  # XXX this flag is mostly for testing...
        error_correction: bool = False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        # if not 0.0 < betas[0] < 1.0:
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay))

        self._quantize = quantize and torch.cuda.is_available()
        self._compress_state_dict = compress_state_dict
        self._error_correction = error_correction
        if error_correction and not _fused:
            raise NotImplementedError(
                "Error correction requires fused kernels.")
        defaults = dict(lr=lr,
                        initial_lr=lr,
                        betas=betas,
                        weight_decay=weight_decay,
                        fused=_fused)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                self.step_param(p, group)

        return loss

    def step_param(self, p: torch.Tensor, hparams: Dict[str, Any]) -> None:
        if not p.requires_grad or p.grad is None:
            return
        if self._quantize and not p.is_cuda:
            raise NotImplementedError(
                f"Can't use quantization with param on {p.device} " +
                f"({p.shape}, {p.dtype}). If you need " +
                "to use DecoupledLionW_8bit without a CUDA device, try " +
                "creating this optimizer with quantize=False.")
        state = self.state[p]  # type:ignore using tensor as key
        if _KEY_MOMENTUM not in state:
            mom = torch.zeros_like(p)
            state[_KEY_MOMENTUM] = _MaybeQuantizedTensor(
                mom, try_quantize=self._quantize)
        need_errs = (p.dtype != torch.float32) and self._error_correction
        if state.get(_KEY_ERRORS) is None and need_errs:
            state[_KEY_ERRORS] = torch.zeros(p.shape,
                                             dtype=torch.uint8,
                                             device=p.device)
        decay_factor = hparams['weight_decay']
        decay_factor *= hparams['lr'] / hparams['initial_lr']
        _lion8b_step(momentums=state[_KEY_MOMENTUM],
                     weights=p,
                     grads=p.grad,
                     beta1=hparams['betas'][0],
                     beta2=hparams['betas'][1],
                     lr=hparams['lr'],
                     weight_decay=decay_factor,
                     fused=hparams['fused'],
                     errors=state.get(_KEY_ERRORS))

    def __setstate__(self, state: Dict[str, Dict[str, Any]]) -> None:
        # we override this function to quantize optimizer states when
        # loading a state dict
        opt_state, _ = state.values()  # other val is param_groups
        for param_id in opt_state:
            param_state = opt_state[param_id]
            new_state = {}
            if _KEY_MOMENTUM in param_state:
                qtensor = _MaybeQuantizedTensor(None,
                                                try_quantize=self._quantize)
                qtensor.load_state_dict(param_state[_KEY_MOMENTUM])
                new_state[_KEY_MOMENTUM] = qtensor
            if self._error_correction and _KEY_ERRORS in param_state:
                # we need to cast back to the correct dtype since optimizer
                # load_state_dict casts to param dtype for fp params; see
                # https://github.com/pytorch/pytorch/blob/a25eee1d77d93079614fab3ea4ac66e64fb2343b/torch/optim/optimizer.py#L626C7-L626C7 # noqa
                errs = param_state[_KEY_ERRORS].to(dtype=torch.uint8)
                new_state[_KEY_ERRORS] = errs
            opt_state[param_id] = new_state
        super().__setstate__(state)

    def state_dict(self):
        # If the user hasn't opted into storing compressed state dicts
        # we have to make sure our states are regular torch.Tensors. This
        # is mostly needed to make FSDP happy in the case that we want to
        # resume training with a number of devices where
        #   (param numel / device count) % quantization group size != 0
        # for any param.
        d = super().state_dict()
        opt_state, _ = d.values()  # other val is param_groups
        for param_id in opt_state:
            # make a copy so that we don't mutate our self.state; opt_state
            # isn't the same as self.state, but its consituent dicts are
            # the same as those in self.state
            param_state = {k: v for k, v in opt_state[param_id].items()}
            if _KEY_MOMENTUM in param_state:
                qtensor = param_state[_KEY_MOMENTUM]
                assert isinstance(qtensor, _MaybeQuantizedTensor)  # pyright
                param_state[_KEY_MOMENTUM] = qtensor.state_dict(
                    allow_quantized=self._compress_state_dict)
            opt_state[param_id] = param_state
        return d


class _MaybeQuantizedTensor:
    """Helper class so 8b LION doesn't have to know quantization details.

    Important points about this class:
    * It handles CPU tensors not being quantized
    * It knows how to save + load state dicts, handling both the quantized
        and not quantized cases
    * It implements some parts of the torch.Tensor interface that we need,
        but is not intended to be a full torch.Tensor replacement
    """

    def __init__(self, data: Optional[torch.Tensor], try_quantize: bool = True):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.quantized: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self._try_quantize = try_quantize and torch.cuda.is_available()

        # conditionally import CUDA kernels
        self._f_encode = None
        self._f_decode = None
        if self._try_quantize:
            from turbo import dequantize8b, quantize8b
            self._f_encode = quantize8b
            self._f_decode = dequantize8b

        if data is not None:
            self.set_data(data)

    def state_dict(self,
                   allow_quantized: bool = False) -> Dict[str, torch.Tensor]:
        if self.is_quantized() and allow_quantized:
            assert self.quantized is not None  # pyright
            assert self.scales is not None  # pyright
            return {'quantized': self.quantized, 'scales': self.scales}
        return {'data': self.materialize().to(dtype=torch.bfloat16)}

    def load_state_dict(self, d: Dict[str, torch.Tensor]) -> None:
        if 'data' in d:
            if len(d) != 1:
                raise ValueError('If state dict specifies "data", it must not' +
                                 f'specify other keys. Got {list(d.keys())}')
            self.set_data(d['data'])
            return

        self.quantized = d['quantized'].to(dtype=torch.int8)
        self.scales = d['scales'].to(dtype=torch.float16)

    def set_data(self, data: torch.Tensor) -> None:
        if not (self._try_quantize and data.is_cuda):
            self.data = data.to(dtype=torch.float32)
            self.quantized = None
            self.scales = None
        else:
            self.data = None
            assert self._f_encode is not None  # pyright
            self.quantized, self.scales = self._f_encode(data)

    def is_quantized(self) -> bool:
        return self.data is None

    def materialize(self) -> torch.Tensor:
        if not self.is_quantized():
            assert self.data is not None  # pyright
            return self.data
        assert self._f_decode is not None  # pyright
        assert self.quantized is not None  # pyright
        assert self.scales is not None  # pyright
        return self._f_decode(self.quantized, self.scales)

    @property  # property to mirror Tensor interface
    def is_cuda(self) -> bool:
        if self.is_quantized():
            assert self.quantized is not None  # pyright
            return self.quantized.is_cuda
        assert self.data is not None  # pyright
        return self.data.is_cuda

    @property  # property to mirror Tensor interface
    def shape(self) -> Tuple[int]:
        if self.is_quantized():
            assert self.quantized is not None  # pyright
            return self.quantized.shape
        assert self.data is not None  # pyright
        return self.data.shape

    def numel(self) -> int:
        if self.is_quantized():
            assert self.quantized is not None  # pyright
            return self.quantized.numel()
        assert self.data is not None  # pyright
        return self.data.numel()

    def __repr__(self):
        return (f'{self.__class__.__name__} quantized={self.is_quantized()} ' +
                f'shape={self.shape}')


def lion_step_unfused(grads: torch.Tensor,
                      weights: torch.Tensor,
                      momentums: torch.Tensor,
                      lr: float,
                      beta1: float,
                      beta2: float,
                      weight_decay: float = 0) -> torch.Tensor:
    # f32 cast to match fused impl + for compatibility with f32 grads or weights
    momentums = momentums.to(torch.float32)
    grads = grads.to(dtype=torch.float32)

    update = momentums.lerp(grads, 1 - beta1).sign_()
    if weight_decay > 0:
        weights.mul_(1. - weight_decay)

    weights.add_(update, alpha=-lr)
    momentums.lerp_(grads, 1. - beta2)
    return momentums  # f32 upcast means not necessarily modified in place


def _lion8b_step(grads: torch.Tensor,
                 weights: torch.Tensor,
                 momentums: _MaybeQuantizedTensor,
                 lr: float,
                 beta1: float,
                 beta2: float,
                 weight_decay: float = 0,
                 errors: Optional[torch.Tensor] = None,
                 fused: bool = True) -> None:

    if momentums.is_quantized() and fused:
        from turbo import lion8b_step as lion8b_step_fused

        assert momentums.quantized is not None  # pyright
        assert momentums.scales is not None  # pyright
        return lion8b_step_fused(grads=grads,
                                 weights=weights,
                                 momentums=momentums.quantized,
                                 scales=momentums.scales,
                                 lr=lr,
                                 beta1=beta1,
                                 beta2=beta2,
                                 weight_decay=weight_decay,
                                 errors=errors)

    momentums_float = momentums.materialize()
    new_momentums = lion_step_unfused(grads=grads,
                                      weights=weights,
                                      momentums=momentums_float,
                                      lr=lr,
                                      beta1=beta1,
                                      beta2=beta2,
                                      weight_decay=weight_decay)
    momentums.set_data(new_momentums)
