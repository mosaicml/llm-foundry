from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader

from llmfoundry.utils.builders import build_tokenizer
from omegaconf import DictConfig
from composer.utils import reproducibility

from llmfoundry.data.packing import BinPackDataset



# def test_simple_packing():
#     # TODO: write simple test
#     # TODO: investigate base version, do outputs match okay?
#     tokenizer = build_tokenizer('mosaicml/mpt-7b', {})
#     BinPackDataset(
#         dataset,
#         packing_ratio, 
#         target_batch_size,
#         10, tokenizer.pad_token_id,
#         'left',
#     )

def test_auto_packing():
    reproducibility.seed_all(17)
    dataloader_cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': 'mosaicml/dolly_hhrlhf',
            'split': 'train',
            'max_seq_len': 1024,
            'allow_pad_trimming': False,
            'decoder_only_format': True,
            'packing_ratio': 'auto',
            'shuffle': False,
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

    print(next(iter(dataloader)))

    # print('length!', len([sample for sample in dataloader]))

    # dataloader_cfg = DictConfig({
    #     'name': 'finetuning',
    #     'dataset': {
    #         'hf_name': 'mosaicml/dolly_hhrlhf',
    #         'split': 'train',
    #         'max_seq_len': 1024,
    #         'allow_pad_trimming': False,
    #         'decoder_only_format': True,
    #         'shuffle': False,
    #     },
    #     'drop_last': False,
    #     'num_workers': 1,
    #     'pin_memory': False,
    #     'prefetch_factor': 1,
    #     'persistent_workers': True,
    #     'timeout': 0,
    # })

    # tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

    # dataloader = build_finetuning_dataloader(dataloader_cfg, tokenizer, 6)

    # print(len(dataloader))

    
    # for sample in dataloader:
    #     print(sample)