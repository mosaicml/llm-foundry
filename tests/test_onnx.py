# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
from composer.utils import reproducibility
from transformers import AutoConfig, AutoModelForCausalLM

from llmfoundry import MPTConfig, MPTForCausalLM


def gen_random_batch(batch_size: int, vocab_size: int, max_seq_len: int):
    # generate input batch of random data
    batch = {
        'input_ids':
            torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, max_seq_len),
                dtype=torch.int64,
            ),
        'attention_mask':
            torch.ones(size=(batch_size, max_seq_len), dtype=torch.bool)
    }
    return batch


def test_onnx_export(tmp_path):
    reproducibility.seed_all(42)
    AutoConfig.register('mpt', MPTConfig)
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)

    batch_size, vocab_size, max_seq_len = 1, 50368, 128

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=64,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=max_seq_len,
        emb_pdrop=0.0,
        resid_pdrop=0.0,
        attn_config={
            'attn_impl': 'torch',
            'alibi': True,
        },
        use_cache=True,
        vocab_size=vocab_size,
        norm_type='layernorm',
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()

    print('Creating random batch...')
    sample_input = gen_random_batch(batch_size, vocab_size, max_seq_len)

    with torch.no_grad():
        mpt(**sample_input)

    torch.onnx.export(
        mpt,
        (sample_input,),
        str(tmp_path / 'mpt.onnx'),
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        opset_version=16,
    )

    with torch.no_grad():
        orig_out = mpt(**sample_input)

    import onnx  # type: ignore
    import onnx.checker  # type: ignore
    import onnxruntime as ort  # type: ignore

    _ = onnx.load(str(tmp_path / 'mpt.onnx'))

    onnx.checker.check_model(str(tmp_path / 'mpt.onnx'))

    ort_session = ort.InferenceSession(str(tmp_path / 'mpt.onnx'))

    for key, value in sample_input.items():
        sample_input[key] = value.cpu().numpy()

    loaded_model_out = ort_session.run(None, sample_input)

    torch.testing.assert_close(
        orig_out.logits.detach().numpy(),
        loaded_model_out[0],
        rtol=1e-4,
        atol=1e-4,
        msg=f'output mismatch between the orig and onnx exported model',
    )
