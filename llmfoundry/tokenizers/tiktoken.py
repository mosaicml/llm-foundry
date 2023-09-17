# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class TiktokenTokenizerWrapper(PreTrainedTokenizer):
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self,
                 model_name: Optional[str] = None,
                 encoding_name: Optional[str] = None,
                 add_bos_token: bool = False,
                 unk_token: Optional[str] = '<|endoftext|>',
                 eos_token: Optional[str] = '<|endoftext|>',
                 bos_token: Optional[str] = '<|endoftext|>',
                 pad_token: Optional[str] = None,
                 **kwargs: Dict[str, Any]):
        try:
            import tiktoken
        except:
            raise ImportError(
                'You need to install tiktoken to use TiktokenTokenizerWrapper.')

        if model_name is None and encoding_name is None:
            raise ValueError(
                'You need to specify either model_name or encoding_name.')

        self.model_name = model_name
        self.encoding_name = encoding_name

        if self.model_name is not None:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        elif self.encoding_name is not None:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        else:
            raise ValueError(
                'You need to specify either model_name or encoding_name.')

        self.add_bos_token = add_bos_token

        super().__init__(model_name=model_name,
                         encoding_name=encoding_name,
                         add_bos_token=add_bos_token,
                         unk_token=unk_token,
                         eos_token=eos_token,
                         bos_token=bos_token,
                         pad_token=None,
                         **kwargs)

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self.encoding.n_vocab

    @property
    def is_fast(self) -> bool:
        return False

    def get_vocab(self):
        """Returns vocab as a dict."""
        vocab = {}
        for i in range(self.vocab_size):
            try:
                # need to try this first, so that we get a proper KeyError,
                # otherwise it crashes in the rust code
                _ = self.encoding.decode_single_token_bytes(i)
                vocab[self.encoding.decode([i])] = i
            except KeyError:
                pass

        return vocab

    def _tokenize(self, text: str, **kwargs: Dict[str, Any]):
        """Returns a tokenized string."""
        tokens = [
            self._convert_id_to_token(t)
            for t in self.encoding.encode(text, allowed_special='all')
        ]

        return tokens

    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab."""
        return self.encoding.encode(token, allowed_special='all')[0]

    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.encoding.decode([index])

    def convert_tokens_to_string(self, tokens: List[str]):
        """Converts a sequence of tokens (string) in a single string."""
        return ''.join(tokens)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """Retrieves sequence ids from a token list that has no special tokens.

        added. This method is called when adding special tokens using the
        tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=False)

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Create a mask from the two sequences passed to be used in a sequence-

        pair classification task. A FAIRSEQ.

        Transformer sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str] = None) -> Tuple[str]:

        # ignore the below type to keep the original signature
        # we are knowingly breaking the signature here, although not 100% certain
        # it doesn't have side effects
        return (None, None)  # type: ignore


TiktokenTokenizerWrapper.register_for_auto_class()
