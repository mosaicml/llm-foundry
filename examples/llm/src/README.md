# Using Composer + FSDP

With the 0.11.0 release of Composer, we have integrated PyTorch's [FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html) engine with some syntactic sugar to make it easy to write custom models that work with Composer + FSDP.

## How the Trainer prepares your model for FSDP
At a high level, when you use the Composer Trainer, you must pass it a `ComposerModel` like [`ComposerMosaicGPT`](./mosaic_gpt.py#L190) that defines certain functions like `forward`, `eval_forward`, `loss`, etc. that are called during the training loop.

Inside that `ComposerModel` you may have one or many submodules, such as a `.model` or `.language_model` or `.classifier` that is the actual `torch.nn.Module` that you will be deploying at inference time. In our case, this is the [`MosaicGPT`](./mosaic_gpt.py#L106) module that we build and attach `ComposerMosaicGPT.model`.

When you provide an `fsdp_config={...}` dictionary to the Composer Trainer, then on `__init__`, the Trainer will attempt to wrap **each of the submodules** of your `ComposerModel` with an FSDP auto wrap policy. This wrapping is recursive, so not only is `MosaicGPT` wrapped, but all submodules of `MosaicGPT` may/may not be wrapped too. See the [FSDP documentation](https://pytorch.org/docs/stable/fsdp.html) for more details on how auto wrap policies work.


## Composer's FSDP Auto Wrap Policy
To make auto-wrapping easier on users, Composer uses a custom auto wrap policy that wraps modules according to the following rules:
1) If any module is attributed with `module._fsdp_wrap = True | False`, that choice will be respected.
2) If the root module (e.g. `MosaicGPT`) defines a function `def fsdp_wrap_fn(module: torch.nn.Module) -> bool`, then that function will be used to evaluate the root module's children.
3) If any module has more parameters than `fsdp_config['min_params']`, it will be wrapped.

These rules are meant to make it easy for users to modify existing models for usage with FSDP. You can either add attributes to modules you want to wrap (#1), define a filter (#2), or make no changes at all and just use the size-based policy via `fsdp_config['min_params'] = ...` (#3).

In `mosaic_gpt.py`, you can see that [we used rule #2](./mosaic_gpt.py#L182) to specify that all `GPTBlock` modules within `GPT` should be wrapped. Alternatively, we could have easily attributed each of the blocks with `block._fsdp_wrap = True` and it would have accomplished the same thing. Whatever style you prefer, it's up to you!

A very similar auto wrap policy is provided for activation checkpointing, with analgous rule #1 that looks for `module._activation_checkpointing = True | False` and rule #2 that looks for `def activation_checkpointing_fn(module: torch.nn.Module) -> bool`.

## The FSDP Config
The full spec and defaults for Composer's `fsdp_config` is here:
```python
fsdp_config = {
  'sharding_strategy': str = 'FULL_SHARD' | 'SHARD_GRAD_OP' | 'NO_SHARD', # Default: 'FULL_SHARD'
  'min_params': float # Default: 1e8
  'cpu_offload': bool = True | False, # Default: False, cpu_offload not supported yet
  'mixed_precision': str = 'FULL' | 'DEFAULT' | 'PURE', # Default: 'DEFAULT'
  # Note: you can explictly provide a dictionary too
  # 'mixed_precision': dict = {
  #   'param_dtype': 'fp32' | 'fp16' | 'bf16',
  #   'reduce_dtype': 'fp32' | 'fp16' | 'bf16',
  #   'buffer_dtype': 'fp32' | 'fp16' | 'bf16',
  # },
  'backward_prefetch': str = 'BACKWARD_PRE' | 'BACKWARD_POST' | 'NONE', # Default: 'BACKWARD_POST'
  'activation_checkpointing': bool = True | False, # Default: False
  'activation_cpu_offload': bool = True | False, # Default: False
}
```

All values come with defaults and can be optionally defined in the `fsdp_config`. Most parameters map directly to parameters in the [FSDP documentation](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel).

One Composer-specific pattern is that if `mixed_precision` is provided as a `str`, then we automatically infer the settings to use from the Trainer's `precision`, which is already being used for autocast, and we construct an associated MixedPrecision object for FSDP:

```python
# If mixed_precision = 'FULL'
mixed_precision = MixedPrecision(
  param_dtype=torch.float32,
  reduce_dtype=torch.float32,
  buffer_dtype=torch.float32,
)
# If mixed_precision = 'DEFAULT'
mixed_precision = MixedPrecision(
  param_dtype=torch.float32,
  reduce_dtype=autocast_precision, # Low precision gradient communication
  buffer_dtype=torch.float32,
)

# If mixed_precision = 'PURE'
mixed_precision = MixedPrecision(
  param_dtype=autocast_precision, # Low precision master weights
  reduce_dtype=autocast_precision, # Low precision gradient communication
  buffer_dtype=autocast_precision, # Low precision buffers
)
```

Thanks for reading this far!
