# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

############################################
# This script takes a previously trained checkpoint, replaces flash/triton attn with torch attention and exports the
# model in onnx format. Checkpoint and exported model paths can be object store.
#
# You can specify the same config as training but some parameters (load_path) are required.
#
# Note: please ignore the message about missing keys in the checkpoint. This is due to the name of params being
# different in flash/triton vs torch causal attention.
# Example usage:
#
#    1) Local checkpoint + local export
#
#    python scripts/export_for_inference.py yamls/mosaic_gpt/1b.yaml load_path=ep0-ba20-rank0.pt
#    export_save_path=export_test/model.onnx
#
#    2) To verify the exported model
#
#    python scripts/export_for_inference.py yamls/mosaic_gpt/1b.yaml load_path=ep0-ba20-rank0.pt
#    export_save_path=export_test/model.onnx verify_exported_model=True
#
#    3) remote checkpoint + local export
#
#    python scripts/export_for_inference.py yamls/mosaic_gpt/1b.yaml
#    load_path=s3://my-bucket/my-folder/gpt-1b/checkpoints/latest-rank{rank}.pt export_save_path=model.onnx
#
#    4) remote checkpoint + remote export
#
#    python scripts/export_for_inference.py yamls/mosaic_gpt/1b.yaml
#    load_path=s3://my-bucket/my-folder/gpt-1b/checkpoints/latest-rank{rank}.pt
#    export_save_path=s3://my-bucket/my-folder/gpt-1b/exported/model.onnx
############################################
import contextlib
import os
import sys
import tempfile
import warnings
from urllib.parse import urlparse

import torch
from composer import Trainer
from composer.utils import get_device, maybe_create_object_store_from_uri
from omegaconf import OmegaConf as om

from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY


def gen_random_batch(batch_size, cfg):
    # generate input batch of random data
    batch = {
        'input_ids':
            torch.randint(
                low=0,
                high=cfg.model.vocab_size,
                size=(batch_size, cfg.max_seq_len),
                dtype=torch.int64,
            ),
        'attention_mask':
            torch.ones(size=(batch_size, cfg.max_seq_len), dtype=torch.int64)
    }
    return batch


def build_composer_model(cfg):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property',
    )
    try:
        return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)
    except:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def main(cfg):
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        f'torch.distributed.*_base is a private function and will be deprecated.*'
    )
    warnings.filterwarnings(action='ignore',
                            category=UserWarning,
                            message='No optimizer was specified.*')
    warnings.filterwarnings(action='ignore',
                            category=UserWarning,
                            message='You are using cfg.init_device=.*')

    init_device = cfg.model.get('init_device', 'cpu')
    if init_device == 'meta':
        warnings.warn('Changing init_device to cpu for export!!')
        cfg.model.init_device = 'cpu'

    load_path = cfg.get('load_path', None)
    if load_path is None:
        raise ValueError('Checkpoint load_path is required for exporting.')

    export_batch_size = 1
    if cfg.get('export_batch_size', None):
        export_batch_size = cfg.export_batch_size
    else:
        warnings.warn(f'Using a batch size of {export_batch_size} for export!!')

    # Build Model
    print('Initializing model...')
    orig_model = build_composer_model(cfg.model)

    # Loading checkpoint using Trainer
    print('Loading model weights...')
    trainer = Trainer(model=orig_model,
                      load_path=load_path,
                      load_weights_only=True)
    # load export model with torch attention
    attn_impl = cfg.model.get('attn_impl', 'torch')
    if attn_impl == 'triton' or attn_impl == 'flash':
        print(
            f'Replacing {cfg.model.attn_impl} attention with torch causal attention'
        )
        cfg.model.attn_impl = 'torch'
        export_model = build_composer_model(cfg.model)
        trainer = Trainer(model=export_model,
                          load_path=load_path,
                          load_weights_only=True)
        # replace flash/triton attention with torch causal attention
        for idx in range(cfg.model.n_layers):
            export_model.model.transformer.blocks[idx].attn.load_state_dict(
                orig_model.model.transformer.blocks[idx].attn.state_dict())
    else:
        export_model = orig_model

    print('model loading done ...')

    export_model.eval()

    cpu_device = get_device('cpu')
    cpu_device.module_to_device(export_model)

    sample_input = gen_random_batch(export_batch_size, cfg)
    # run forward pass on orig model once
    with torch.no_grad():
        export_model(sample_input)

    save_path = cfg.get('export_save_path', 'model.onnx')
    save_object_store = maybe_create_object_store_from_uri(save_path)
    is_remote_store = save_object_store is not None
    tempdir_ctx = (tempfile.TemporaryDirectory()
                   if is_remote_store else contextlib.nullcontext(None))
    with tempdir_ctx as tempdir:
        if is_remote_store:
            local_save_path = os.path.join(str(tempdir), 'model.onnx')
        else:
            local_save_path = save_path
        trainer.export_for_inference(
            save_format='onnx',
            save_path=local_save_path,
            sample_input=sample_input,
        )

        if cfg.get('verify_exported_model', False):
            test_input = gen_random_batch(export_batch_size, cfg)
            with torch.no_grad():
                orig_out = export_model(test_input)

            import onnx  # type: ignore
            import onnx.checker  # type: ignore
            import onnxruntime as ort  # type: ignore

            loaded_model = onnx.load(local_save_path)

            onnx.checker.check_model(loaded_model)

            ort_session = ort.InferenceSession(local_save_path)

            for key, value in test_input.items():
                test_input[key] = value.cpu().numpy()

            loaded_model_out = ort_session.run(None, test_input)

            torch.testing.assert_close(
                orig_out.detach().numpy(),
                loaded_model_out[0],
                rtol=1e-4,  # lower tolerance for ONNX
                atol=1e-3,  # lower tolerance for ONNX
                msg=f'output mismatch between the orig and onnx exported model',
            )
            print('exported model ouptut matches with unexported model!!')

        # upload if required.
        if is_remote_store:
            remote_path = urlparse(save_path).path
            save_object_store.upload_object(remote_path.lstrip('/'),
                                            local_save_path)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
