# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from contextlib import nullcontext
from functools import partial
from typing import List, Optional

import pytest
import shutil
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed._tensor import DTensor, Placement, Replicate, Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict)
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.nn.parallel import DistributedDataParallel as DDP

from llmfoundry.models.layers.dmoe import dMoE
from llmfoundry.models.layers.ffn import dtensorify_param
from llmfoundry.models.mpt.configuration_mpt import MPTConfig
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM

try:
    import megablocks
    is_megablocks_imported = True
except ModuleNotFoundError:
    is_megablocks_imported = False


def _get_all_inputs(
    input_shape: List[int],
    dtype: Optional[torch.dtype],
):
    world_size: int = dist.get_world_size()
    rank: int = dist.get_rank()
    device: torch.device = torch.device(f'cuda:{rank}')
    all_inputs = []
    for _ in range(world_size):
        all_inputs.append(torch.rand(
            input_shape,
            device=device,
            dtype=dtype,
        ))
    return all_inputs


def _get_torch_dtype(fp16: bool, bf16: bool) -> Optional[torch.dtype]:
    if fp16:
        return torch.float16
    elif bf16:
        return torch.bfloat16
    return None


@pytest.mark.skipif(not is_megablocks_imported,
                    reason='This test needs megablocks module')
@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize('moe_num_experts', [8])
@pytest.mark.parametrize('mlp_type', ['glu', 'mlp'])
@pytest.mark.parametrize('moe_world_size', [1, 2])
@pytest.mark.parametrize('two_d_input', [True, False])
def test_dmoe(moe_num_experts: int, mlp_type: str, moe_world_size: int,
              two_d_input: bool):
    # Generate inputs
    rank = dist.get_rank()
    batch_size = 2
    seq_len = 3
    hidden_size = 128
    if two_d_input:
        input_shape = [batch_size * seq_len, hidden_size]
    else:
        input_shape = [batch_size, seq_len, hidden_size]
    fp16 = False
    bf16 = True
    dtype = _get_torch_dtype(fp16, bf16)
    x = _get_all_inputs(input_shape, dtype)[rank]

    # Construct DDP torch dMoE
    device = torch.device(f'cuda:{dist.get_rank()}')
    common_args = {
        'hidden_size': hidden_size,
        'ffn_hidden_size': hidden_size,
        'moe_top_k': 2,
        'activation_fn': partial(F.gelu, approximate='none'),
        'moe_jitter_eps': 0.0,  # Disable randomiztion
        'moe_normalize_expert_weights': 1,
        'uniform_expert_assignment': False,
        'bias': False,
        'device': device,
        'moe_num_experts': moe_num_experts,
        'mlp_type': mlp_type,
    }

    torch_dmoe = dMoE(**common_args).to(device, dtype=dtype)
    torch_dmoe = DDP(
        torch_dmoe,
        device_ids=[rank],
    )
    torch_dmoe_optimizer = optim.SGD(torch_dmoe.parameters(), lr=0.1)

    # Construct TP MB dMoE
    mp_dmoe_args = copy.deepcopy(common_args)
    extra_args = {
        'fp16': fp16,
        'bf16': bf16,
        'init_method': partial(torch.nn.init.uniform_, a=-1.0, b=1.0),
    }
    device_mesh = None
    if moe_world_size > 1:
        world_size = dist.get_world_size()
        assert world_size % moe_world_size == 0
        moe_dp_dim = world_size // moe_world_size
        device_mesh = init_device_mesh(
            'cuda',
            (moe_dp_dim, moe_world_size),
            mesh_dim_names=('weight_parallel', 'expert_parallel'),
        )
        expert_parallel_group = device_mesh['expert_parallel'].get_group(0)
        extra_args.update(
            {
                'moe_expert_model_parallelism': True,
                'expert_parallel_group': expert_parallel_group,
            },)
    mp_dmoe_args.update(extra_args)
    args = megablocks.layers.arguments.Arguments(**mp_dmoe_args,)
    mb_dmoe = megablocks.layers.dmoe.dMoE(args).to(device)
    mb_dmoe.router = DDP(mb_dmoe.router, device_ids=[rank])

    if moe_world_size > 1:
        assert device_mesh is not None
        two_d_placements: List[Placement] = [Replicate(), Shard(0)]
        dtensorified_params = [(
            name,
            dtensorify_param(
                param=parameter,
                mesh=device_mesh,
                placements=two_d_placements,
            ),
        ) for name, parameter in mb_dmoe.experts.mlp.named_parameters()]
        tp_names = []
        for name, dtensorified_param in dtensorified_params:
            mb_dmoe.experts.mlp.register_parameter(name, dtensorified_param)
            tp_names.append('experts.mlp.' + name)

        _pre_dp_module_transform(mb_dmoe.experts.mlp)

        dp_pg = device_mesh['weight_parallel'].get_group(0)
        mb_dmoe.experts = DDP(mb_dmoe.experts, process_group=dp_pg)

        # Copy mb_dmoe's parameters to torch_dmoe
        mb_dmoe_state_dict = get_model_state_dict(mb_dmoe,
                                                  options=StateDictOptions(
                                                      full_state_dict=True,))
        for key, t in mb_dmoe_state_dict.items():
            if key in tp_names:
                dtensor_full = DTensor.from_local(
                    t,  # pyright: ignore[reportGeneralTypeIssues]
                    device_mesh=device_mesh,
                    placements=two_d_placements,
                ).full_tensor()

                mb_dmoe_state_dict[key] = dtensor_full
    else:
        mb_dmoe.experts = DDP(mb_dmoe.experts, device_ids=[rank])
        mb_dmoe_state_dict = get_model_state_dict(mb_dmoe,
                                                  options=StateDictOptions(
                                                      full_state_dict=True,))
    mb_dmoe_optimizer = optim.SGD(mb_dmoe.parameters(), lr=0.1)

    # Load mb_dmoe state dict to torch dmoe
    torch_dmoe.module.load_state_dict(mb_dmoe_state_dict, strict=True)

    # Run train_step check
    torch_y = torch_dmoe(x)
    mb_y = mb_dmoe(x)

    torch_y.sum().backward()
    mb_y.sum().backward()
    torch_dmoe_optimizer.step()
    mb_dmoe_optimizer.step()

    torch_y = torch_dmoe(x)
    mb_y = mb_dmoe(x)
    torch.testing.assert_close(torch_y, mb_y)

# TODO(GRT-2435): Change to fixture
def delete_transformers_cache():
    # Only delete the files on local rank 0, otherwise race conditions are created
    if not dist.get_local_rank() == 0:
        return

    hf_cache_home = os.path.expanduser(
        os.getenv(
            'HF_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                         'huggingface')))
    HF_MODULES_CACHE = os.getenv('HF_MODULES_CACHE',
                                 os.path.join(hf_cache_home, 'modules'))
    if os.path.exists(HF_MODULES_CACHE) and os.path.isdir(HF_MODULES_CACHE):
        shutil.rmtree(HF_MODULES_CACHE)

@pytest.mark.skipif(not is_megablocks_imported,
                    reason='This test needs megablocks module')
@pytest.mark.gpu
@pytest.mark.parametrize('seqlen', [512])
@pytest.mark.parametrize('mlp_type', ['glu', 'mlp'])
@pytest.mark.parametrize('precision', ['bf16', 'fp32'])
def test_fwd_equal_dmoe(seqlen: int, precision: str, mlp_type: str):
    delete_transformers_cache()

    mb_dmoe_config = MPTConfig(d_model=1024,
                               n_heads=32,
                               n_layers=1,
                               learned_pos_emb=False,
                               max_seq_len=2048,
                               vocab_size=100,
                               no_bias=True,
                               fuse_norm_attn_norm=True,
                               tie_word_embeddings=False,
                               attn_config=dict(
                                   attn_type='grouped_query_attention',
                                   attn_impl='torch',
                                   attn_pdrop=0.0,
                                   clip_qkv=8.0,
                                   kv_n_heads=8,
                                   rope=True,
                                   rope_theta=10000.0,
                               ),
                               ffn_config=dict(
                                   ffn_type='mb_dmoe',
                                   fc_type='torch',
                                   mlp_type=mlp_type,
                                   moe_world_size=1,
                                   ffn_act_fn={'name': 'silu'},
                                   ffn_hidden_size=1792,
                                   moe_num_experts=16,
                                   moe_top_k=4,
                                   moe_jitter_eps=0.0,
                                   moe_loss_weight=0.05,
                                   moe_normalize_expert_weights=1.0,
                                   uniform_expert_assignment=False,
                               ))
    device = 'cuda:0'
    if precision == 'fp32':
        dtype = torch.float32
        context = nullcontext()
    elif precision == 'bf16':
        dtype = torch.bfloat16
        context = torch.autocast('cuda', torch.bfloat16)
    else:
        raise ValueError(f'Invalid {precision=}')

    torch_dmoe_config = copy.deepcopy(mb_dmoe_config)
    torch_dmoe_config.ffn_config['ffn_type'] = 'torch_dmoe'

    mb_dmoe_model = MPTForCausalLM(mb_dmoe_config).to(device=device,
                                                      dtype=dtype)
    torch_dmoe_model = MPTForCausalLM(torch_dmoe_config).to(device=device,
                                                            dtype=dtype)

    # set same state dicts
    torch_dmoe_model.load_state_dict(mb_dmoe_model.state_dict())

    # tokens
    token_ids = torch.randint(
        0,
        mb_dmoe_config.vocab_size,
        (1, seqlen),
        device=device,
        dtype=torch.long,
    )

    with context:
        mpt_logits = mb_dmoe_model(token_ids).logits
        db_logits = torch_dmoe_model(token_ids).logits
        assert torch.allclose(mpt_logits, db_logits, rtol=0.01, atol=0.01)

    delete_transformers_cache()
