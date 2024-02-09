# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY


def pass_on_block_idx(parent: torch.nn.Module):
    if not hasattr(parent, 'block_idx') or not hasattr(parent, 'max_block_idx'):
        return
    for child in parent.children():
        child.block_idx = parent.block_idx
        child.max_block_idx = parent.max_block_idx
        if child.children():
            pass_on_block_idx(child)


def get_act_ckpt_module(mod_name: str) -> Any:
    """Get the module type from the module name."""
    if mod_name.lower() == 'mptblock':
        mod_type = MPTBlock
    elif mod_name in ATTN_CLASS_REGISTRY:
        mod_type = ATTN_CLASS_REGISTRY[mod_name]
    elif mod_name in FFN_CLASS_REGISTRY:
        mod_type = FFN_CLASS_REGISTRY[mod_name]
    elif mod_name in NORM_CLASS_REGISTRY:
        mod_type = NORM_CLASS_REGISTRY[mod_name]
    else:
        msg = ', '.join(
            list(ATTN_CLASS_REGISTRY.keys()) + list(FFN_CLASS_REGISTRY.keys()) +
            list(NORM_CLASS_REGISTRY.keys()) + ['MPTBlock'])
        raise ValueError(
            f'{mod_name} (specified in activation_checkpointing_target) is not a recognized option out of available options {msg}.'
        )
    return mod_type


def parse_ele_str(ele: str, max_block_idx: int) -> list:
    """Parse a string in target_blocks and return a list of block ids to add.

    Supported formats are: first-n, middle-m, last-k, range-i-j which correspond
    to the first n, the middle m,  the last k, and the range [i, j).
    """
    to_add = None
    if ele.startswith('first-'):
        assert ele[6:].isdigit(), f'Invalid target_blocks element {ele}'
        to_add = list(range(min(int(ele[6:]), max_block_idx + 1)))
    elif ele.startswith('last-'):
        assert ele[5:].isdigit(), f'Invalid target_blocks element {ele}'
        to_add = list(
            range(max(max_block_idx - int(ele[5:]) + 1, 0), max_block_idx + 1))
    elif ele.startswith('middle-'):
        assert ele[7:].isdigit(), f'Invalid target_blocks element {ele}'
        num = int(ele[7:])
        start = max(max_block_idx // 2 - num // 2, 0)
        end = min(start + num, max_block_idx + 1)
        to_add = list(range(start, end))
    elif ele.startswith('range-'):
        r = ele[6:].split('-')
        assert len(r) == 2, f'Invalid target_blocks element {ele}'
        start, end = int(r[0]), int(r[1])
        start = max(start, 0)
        end = min(end, max_block_idx + 1)
        to_add = list(range(start, end))
    else:
        raise ValueError(f'Invalid target_blocks element {ele}')
    return to_add


def get_target_block_list(target_blocks: Any, max_block_idx: int) -> list:
    """Parse the user input and return a list of block ids."""
    candidate_block_ids = []
    if isinstance(target_blocks, int):
        candidate_block_ids = list(range(target_blocks))
    elif isinstance(target_blocks, list):
        for ele in target_blocks:
            if isinstance(ele, int):
                candidate_block_ids.append(ele)
            elif isinstance(ele, str):
                to_add = parse_ele_str(ele, max_block_idx)
                candidate_block_ids.extend(to_add)
            else:
                raise ValueError(
                    f'target_blocks must be a list of integers or "first-n", "middle-m", "last-k", or "range-i-j" where n, m, k, i, j are integers, but got {target_blocks}'
                )
    elif isinstance(target_blocks, str):
        target_blocks = target_blocks.replace(' ', '')
        for ele in target_blocks.split(','):
            to_add = parse_ele_str(ele, max_block_idx)
            candidate_block_ids.extend(to_add)
    else:
        raise ValueError(
            f'target_blocks must be either a single intege, or a list of integers, or a comma separated string made of "first-n", "last-m", "middle-k", "range-i-j", or a list of mixed integers and before-mentioned strings, but got {type(target_blocks)}'
        )

    candidate_block_ids = list(set(candidate_block_ids))
    return candidate_block_ids


def check_mapping_blocks_overlap(mapping: dict, max_block_idx: int) -> None:
    """Check if the block ids in the mapping overlap with each other."""
    all_blocks = [None] * (max_block_idx + 1)
    for k, v in mapping.items():
        if v == -1:
            v = list(range(max_block_idx + 1))
        for vv in v:
            if vv < 0 or vv > max_block_idx:
                continue
            else:
                if all_blocks[vv] is not None:
                    raise ValueError(
                        f'Block {vv} is assigned to both {k} and {all_blocks[vv]}.'
                    )
                else:
                    all_blocks[vv] = k


def build_act_ckpt_mod_to_blocks(act_ckpt_target: Any, top_module: Any,
                                 max_block_idx: int) -> dict:
    act_ckpt_mod_to_blocks = {}
    if act_ckpt_target is None or act_ckpt_target == []:
        mod = top_module
        act_ckpt_mod_to_blocks[mod] = -1
    elif isinstance(act_ckpt_target, str):
        mod = get_act_ckpt_module(act_ckpt_target)
        act_ckpt_mod_to_blocks[mod] = -1
    elif isinstance(act_ckpt_target, list):
        for target in act_ckpt_target:
            mod = get_act_ckpt_module(target)
            act_ckpt_mod_to_blocks[mod] = -1
    elif isinstance(act_ckpt_target, dict):
        for k, v in act_ckpt_target.items():
            mod = get_act_ckpt_module(k)
            block_ids = get_target_block_list(v, max_block_idx)
            act_ckpt_mod_to_blocks[mod] = block_ids
    else:
        raise ValueError(
            f'activation_checkpointing_target must be either a single string or a list or a dict, but got {type(act_ckpt_target)}'
        )

    return act_ckpt_mod_to_blocks
