
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

_KEY_MOMENTUM = 'exp_avg'
_MIN_QUANTIZE_SIZE = 1024  # has to be at least 1024 for our quantization


class Lion8bit(torch.optim.Optimizer):

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float = 1e-3,
                 betas: Tuple[float] = (0.9, 0.99),
                 l2_penalty: float = 0,
                 weight_decay: float = 0,
                 compress_state_dict: bool = False,
                 quantize: bool = True,
                 fused: bool = True,  # XXX this flag is mostly for testing...
                 ):
        """TODO full docstring
        If compress_state_dict is True, resuming on a different number of
        GPUs using FSDP won't work. However, the optimizer state will take
        just over one byte per parameter instead of two bytes.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self._quantize = quantize
        self._compress_state_dict = compress_state_dict
        defaults = dict(lr=lr,
                        betas=betas,
                        l2_penalty=l2_penalty,
                        weight_decay=weight_decay,
                        fused=fused)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_to_update = [p for p in group['params'] if
                                p.grad is not None and p.requires_grad]
            for p in params_to_update:
                state = self.state[p]
                if len(state) == 0:
                    mom = torch.zeros_like(p)
                    state[_KEY_MOMENTUM] = _MaybeQuantizedTensor(
                        mom, try_quantize=self._quantize)
                momentums = state[_KEY_MOMENTUM]
                _lion_step(momentums=momentums,
                           weights=p,
                           grads=p.grad,
                           beta1=group['betas'][0],
                           beta2=group['betas'][1],
                           lr=group['lr'],
                           l2_penalty=group['l2_penalty'],
                           weight_decay=group['weight_decay'],
                           fused=group['fused'])

        return loss

    def __setstate__(self, state) -> None:
        # we override this function to quantize optimizer states when
        # loading a state dict
        opt_state, param_groups = state.values()
        for param_id in opt_state:
            param_state = opt_state[param_id]
            qtensor = _MaybeQuantizedTensor(None, try_quantize=self._quantize)
            qtensor.load_state_dict(param_state[_KEY_MOMENTUM])
            opt_state[param_id] = {_KEY_MOMENTUM: qtensor}
        super().__setstate__(state)

    def state_dict(self):
        # If the user hasn't opted into storing compressed state dicts
        # we have to make sure our states are regular torch.Tensors. This
        # is mostly needed to make FSDP happy in the case that we want to
        # resume training with a number of devices where
        #   (param numel / device count) % block size != 0
        # for any param.
        d = super().state_dict()
        opt_state, param_groups = d.values()
        for param_id in opt_state:
            # make a copy so that we don't mutate our self.state; opt_state
            # isn't the same as self.state, but its consituent dicts are
            # the same as those in self.state
            param_state = {k: v for k, v in opt_state[param_id].items()}
            qtensor = param_state[_KEY_MOMENTUM]
            assert isinstance(qtensor, _MaybeQuantizedTensor)  # pyright
            param_state[_KEY_MOMENTUM] = qtensor.state_dict(
                allow_quantized=self._compress_state_dict)
            opt_state[param_id] = param_state
        return d


class _MaybeQuantizedTensor:

    def __init__(self, data: torch.Tensor, try_quantize: bool = True):
        super().__init__()
        self.data = None
        self._try_quantize = try_quantize and torch.cuda.is_available()

        # conditionally import CUDA kernels
        self._f_encode = None
        self._f_decode = None
        if self._try_quantize:
            import llmfoundry.optim._quantize_kernels as kernels
            self._f_encode = kernels.encode_signed
            self._f_decode = kernels.decode_signed

        if data is not None:
            self.set_data(data)

    def state_dict(self, allow_quantized: bool = False) -> Dict[str, torch.Tensor]:
        if self.is_quantized() and allow_quantized:
            return dict(quantized=self.quantized,
                        scales=self.scales,
                        scale_scales=self.scale_scales)

        # we convert to bf16 here to save space; we don't just return
        # bf16 when decoding because numba CUDA doesn't know this dtype
        return dict(data=self.materialize().to(dtype=torch.bfloat16))

    def load_state_dict(self, d: Dict[str, torch.Tensor]) -> None:
        if 'data' in d:
            if len(d) != 1:
                raise ValueError('If state dict specifies "data", it must not'
                                 f'specify other keys. Got {list(d.keys())}')
            self.set_data(d['data'])
            return

        self.quantized = d['quantized']
        self.scales = d['scales']
        self.scale_scales = d['scale_scales']

    def set_data(self, data: torch.Tensor) -> None:
        try_quantize = self._try_quantize and data.is_cuda
        if (not try_quantize) or (data.numel() < _MIN_QUANTIZE_SIZE):
            self.data = data.to(dtype=torch.bfloat16)
            self.quantized = None
            self.scales = None
            self.scale_scales = None
        else:
            self.data = None
            self.quantized, self.scales, self.scale_scales = self._f_encode(data)

    def is_quantized(self) -> bool:
        return self.data is None

    def materialize(self) -> torch.Tensor:
        if not self.is_quantized():
            return self.data
        return self._f_decode(self.quantized, self.scales, self.scale_scales)

    @property  # property to mirror tensor interface
    def is_cuda(self) -> bool:
        if self.is_quantized():
            return self.quantized.is_cuda
        return self.data.is_cuda

    @property  # property to mirror tensor interface
    def shape(self) -> Tuple[int]:
        if self.is_quantized():
            return self.quantized.shape
        return self.data.shape

    def numel(self) -> int:
        return self.quantized.numel() if self.is_quantized() else self.data.numel()

    def __repr__(self):
        return (f'{self.__class__.__name__} quantized={self.is_quantized()} ' +
                f'shape={self.shape}')


def _lion_step_unfused(momentums: torch.Tensor,
                       weights: torch.Tensor,
                       grads: torch.Tensor,
                       lr: float,
                       beta1: float,
                       beta2: float,
                       l2_penalty: float = 0,
                       weight_decay: float = 0) -> torch.Tensor:
    if l2_penalty > 0:
        grads = grads + (weights * l2_penalty)

    # f32 cast to match fused impl + for compatibility with f32 grads or weights
    momentums = momentums.to(torch.float32)

    update = momentums.lerp(grads, 1 - beta1).sign_()
    if weight_decay > 0:
        update.add_(weights, alpha=weight_decay)

    weights.add_(update, alpha=-lr)
    momentums.lerp_(grads, 1. - beta2)  # NOTE that we modify momentums in place
    return momentums


def _lion_step(momentums: _MaybeQuantizedTensor,
               weights: torch.Tensor,
               grads: torch.Tensor,
               lr: float,
               beta1: float,
               beta2: float,
               l2_penalty: float = 0,
               weight_decay: float = 0,
               fused: bool = True) -> None:

    if momentums.is_quantized() and fused:
        from llmfoundry.optim._quantize_kernels import lion_step_fused

        # TODO rm this check after testing since kernel will throw anyway
        if not momentums.is_cuda:
            raise NotImplementedError('Quantized LION only supported on GPUs')

        return lion_step_fused(momentums_quantized=momentums.quantized,
                               momentums_scales=momentums.scales,
                               momentums_scale_scales=momentums.scale_scales,
                               weights=weights,
                               grads=grads,
                               lr=lr,
                               beta1=beta1,
                               beta2=beta2,
                               l2_penalty=l2_penalty,
                               weight_decay=weight_decay)

    momentums_float = momentums.materialize()
    new_momentums = _lion_step_unfused(momentums_float,
                                       weights=weights,
                                       grads=grads,
                                       lr=lr,
                                       beta1=beta1,
                                       beta2=beta2,
                                       l2_penalty=l2_penalty,
                                       weight_decay=weight_decay)
    momentums.set_data(new_momentums)
