# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch
from composer.utils import reproducibility
from transformers import AutoConfig, AutoModelForCausalLM

from examples.llm import MosaicGPT, MosaicGPTConfig


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
    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)

    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.0,
        resid_pdrop=0.0,
        attn_impl='torch',
        alibi=True,
        use_cache=True,
        vocab_size=50368,
        low_precision_layernorm=False,
    )
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()

    print('Creating random batch...')
    sample_input = gen_random_batch(
        1,
        50368,
        2048,
    )

    with torch.no_grad():
        mosaic_gpt(**sample_input)

    torch.onnx.export(
        mosaic_gpt,
        (sample_input,),
        str(tmp_path / 'mosaic_gpt.onnx'),
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        opset_version=16,
    )

    with torch.no_grad():
        orig_out = mosaic_gpt(**sample_input)

    import onnx  # type: ignore
    import onnx.checker  # type: ignore
    import onnxruntime as ort  # type: ignore

    _ = onnx.load(str(tmp_path / 'mosaic_gpt.onnx'))

    onnx.checker.check_model(str(tmp_path / 'mosaic_gpt.onnx'))

    ort_session = ort.InferenceSession(str(tmp_path / 'mosaic_gpt.onnx'))

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
