# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import transformers


class LLMTokenizer(ABC):
    tokenizer_name: str
    max_seq_len: int

    def __init__(self, tokenizer_name: str, max_seq_len: int):
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError

    @abstractmethod
    def decode(self, x):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError


class HFTokenizer(LLMTokenizer):
    tokenizer_name: str
    max_seq_len: int
    group_method: str  # type: ignore (reportUninitializedInstanceVariable)
    tokenizer: transformers.AutoTokenizer

    def __init__(self, tokenizer_name: str, max_seq_len: int):
        super().__init__(tokenizer_name, max_seq_len)

        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def encode(self, x):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id


TOKENIZER_REGISTRY = {'hftokenizer': HFTokenizer}
