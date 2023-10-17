from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader

from llmfoundry.utils.builders import build_tokenizer
from omegaconf import DictConfig


def test_auto_packing():
    dataloader_cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': 'mosaicml/dolly_hhrlhf',
            'split': 'train',
            'max_seq_len': 1024,
            'allow_pad_trimming': False,
            'decoder_only_format': True,
            'packing_ratio': 'auto',
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 1,
        'pin_memory': False,
        'prefetch_factor': 1,
        'persistent_workers': True,
        'timeout': 0,
    })

    tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

    dataloader = build_finetuning_dataloader(dataloader_cfg, tokenizer, 6)

    print('length!', len([sample for sample in dataloader]))

    dataloader_cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': 'mosaicml/dolly_hhrlhf',
            'split': 'train',
            'max_seq_len': 1024,
            'allow_pad_trimming': False,
            'decoder_only_format': True,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 1,
        'pin_memory': False,
        'prefetch_factor': 1,
        'persistent_workers': True,
        'timeout': 0,
    })

    tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

    dataloader = build_finetuning_dataloader(dataloader_cfg, tokenizer, 6)

    print(len(dataloader))

    
    # for sample in dataloader:
    #     print(sample)