# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import List

import torch
from composer.core import get_precision_context
from omegaconf import OmegaConf as om

from examples.llm.src import COMPOSER_MODEL_REGISTRY, TOKENIZER_REGISTRY


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class MosaicGPTInference:

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)

        prompt_tokens = [self.tokenizer.encode(x) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.cfg.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), -100).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != -100
        start_pos = min_prompt_size
        for cur_pos in range(start_pos, total_len):
            with torch.no_grad():
                with get_precision_context(self.cfg.get('precision',
                                                        'amp_bf16')):
                    logits = self.model.forward(
                        {'input_ids': tokens[:, :cur_pos]})
            logits = logits[:, -1, :]
            logits = logits.float()
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos],
                                     tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_token_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def build_composer_model(cfg):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    try:
        return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)
    except:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def get_mosaicgpt_tokenizer(checkpoint_yaml_path: str):
    with open(checkpoint_yaml_path) as f:
        cfg = om.load(f)
    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.type](**cfg.tokenizer.args)
    return tokenizer


def get_mosaicgpt_inference_model(checkpoint_yaml_path: str, tokenizer):
    with open(checkpoint_yaml_path) as f:
        cfg = om.load(f)
    # set init_device to cpu for checkpoint loading
    # ToDo: Directly load a checkpoint into 'meta' model
    cfg.model.init_device = 'cpu'
    model = build_composer_model(cfg.model)

    ckpt_load_path = cfg.get('load_path', None)  # type: ignore
    if ckpt_load_path is None:
        raise ValueError('Checkpoint load_path is required for exporting.')

    checkpoint = torch.load(ckpt_load_path, map_location='cpu')

    model.load_state_dict(checkpoint['state']['model'], strict=True)

    model.cuda()
    model.eval()

    generator = MosaicGPTInference(cfg, model, tokenizer)

    return generator
