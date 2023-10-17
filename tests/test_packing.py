from typing import List

from llmfoundry.data.packing import BinPackDataset
from torch.utils.data import IterableDataset

class TestDataset(IterableDataset):
    def __init__(self, data: List[List[int]]):
        super().__init__()
        self.data = data

    def __iter__(self):
        for d in self.data:
            yield {'input_ids': d }

def test_simple_packing():
    dataset = TestDataset([
        [1],
        [2] * 2,
        [8] * 8,
        [9] * 9,
    ])

    packed_dataset = BinPackDataset(
        dataset,
        packing_ratio=2, 
        target_batch_size=2,
        max_seq_len=10, 
        pad_token_id=0,
    )

    packed_samples = [sample['input_ids'] for sample in packed_dataset]

    assert packed_samples[0] == [8] * 8 + [2] * 2
    assert packed_samples[1] == [9] * 9 + [1]

def test_simple_packing_with_leftovers():
    dataset = TestDataset([
        [5] * 5,
        [6] * 6,
        [5] * 5,
        [7] * 7,
    ])

    packed_dataset = BinPackDataset(
        dataset,
        packing_ratio=2, 
        target_batch_size=2,
        max_seq_len=10, 
        pad_token_id=0,
    )

    packed_samples = [sample['input_ids'] for sample in packed_dataset]

    assert packed_samples[0] == [5] * 10
    assert packed_samples[1] == [7] * 7
    assert packed_samples[2] == [6] * 6

# def test_auto_packing():
#     reproducibility.seed_all(17)
#     dataloader_cfg = DictConfig({
#         'name': 'finetuning',
#         'dataset': {
#             'hf_name': 'mosaicml/dolly_hhrlhf',
#             'split': 'train',
#             'max_seq_len': 1024,
#             'allow_pad_trimming': False,
#             'decoder_only_format': True,
#             'packing_ratio': 'auto',
#             'shuffle': False,
#         },
#         'drop_last': False,
#         'num_workers': 1,
#         'pin_memory': False,
#         'prefetch_factor': 1,
#         'persistent_workers': True,
#         'timeout': 0,
#     })

#     tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

#     dataloader = build_finetuning_dataloader(dataloader_cfg, tokenizer, 6)

#     print(next(iter(dataloader)))

#     # print('length!', len([sample for sample in dataloader]))

#     # dataloader_cfg = DictConfig({
#     #     'name': 'finetuning',
#     #     'dataset': {
#     #         'hf_name': 'mosaicml/dolly_hhrlhf',
#     #         'split': 'train',
#     #         'max_seq_len': 1024,
#     #         'allow_pad_trimming': False,
#     #         'decoder_only_format': True,
#     #         'shuffle': False,
#     #     },
#     #     'drop_last': False,
#     #     'num_workers': 1,
#     #     'pin_memory': False,
#     #     'prefetch_factor': 1,
#     #     'persistent_workers': True,
#     #     'timeout': 0,
#     # })

#     # tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

#     # dataloader = build_finetuning_dataloader(dataloader_cfg, tokenizer, 6)

#     # print(len(dataloader))

    
#     # for sample in dataloader:
#     #     print(sample)