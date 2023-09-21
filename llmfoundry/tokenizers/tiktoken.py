# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer


class TiktokenTokenizerWrapper(PreTrainedTokenizer):
    """A thin wrapper around tiktoken to make it compatible with Hugging Face.

    tokenizers.

    See HuggingFace for further documentation on general tokenizer methods.
    """

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
        """Constructor creates a tiktoken tokenizer to use as the underlying.

        tokenizer.

        Args:
            model_name (Optional[str], optional): The name of the model to load from tiktoken. Defaults to None.
                Either model_name or encoding_name must be set, but not both.
            encoding_name (Optional[str], optional): The name of the encoding to load from tiktoken. Defaults to None.
                Either model_name or encoding_name must be set, but not both.
            add_bos_token (bool, optional): Whether to add bos tokens. Defaults to False.
            unk_token (Optional[str], optional): The unk token. Defaults to '<|endoftext|>'.
            eos_token (Optional[str], optional): The eos token. Defaults to '<|endoftext|>'.
            bos_token (Optional[str], optional): The bos token. Defaults to '<|endoftext|>'.
            pad_token (Optional[str], optional): The pad token. Defaults to None.
        """
        try:
            import tiktoken
        except:
            raise ImportError(
                'You need to install tiktoken to use TiktokenTokenizerWrapper.')

        if model_name is not None and encoding_name is not None:
            raise ValueError(
                'You need to specify either model_name or encoding_name, not both.'
            )

        self.model_name = model_name
        self.encoding_name = encoding_name

        if self.model_name is not None:
            self.encoding = tiktoken.encoding_for_model(  # type: ignore (thirdParty)
                self.model_name)
        elif self.encoding_name is not None:
            self.encoding = tiktoken.get_encoding(  # type: ignore (thirdParty)
                self.encoding_name)
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
                         pad_token=pad_token,
                         **kwargs)

    @property
    def vocab_size(self) -> int:
        """Returns vocab size."""
        return self.encoding.n_vocab

    @property
    def is_fast(self) -> bool:
        return False

    def get_vocab(self) -> Dict[str, int]:
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

    def _tokenize(self, text: str) -> List[int]:
        """Returns a tokenized string.

        Note: We have slightly redefined the expected contract between this method and
        the _convert_token_to_id method. Normally, this method turns a string, into a list of strings,
        and then the _convert_token_to_id method turns that list of strings into a list of integers.
        However, not all vocab indices can be decoded into a string, so instead we just return the integers
        from this function, and have adjusted the _convert_token_to_id method to handle integers as well as strings.
        The only use of _tokenize that I could find was in this way, so this _should_ be safe.
        """
        if not isinstance(text, str):
            raise ValueError(
                f'Expected a string input to _tokenize but got {type(text)}.')

        tokens = [t for t in self.encoding.encode(text, allowed_special='all')]

        return tokens

    def _convert_token_to_id(self, token: Union[int, str]) -> int:
        """Converts a token (str) into an id using the vocab."""
        if isinstance(token, int):
            return token

        return self.encoding.encode(token, allowed_special='all')[0]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) into a token (str) using the vocab."""
        return self.encoding.decode([index])

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        return ''.join(tokens)

    def convert_ids_to_tokens(
            self,
            ids: Union[int, List[int]],
            skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """Converts a single index or a sequence of indices into a token or a.

        sequence of tokens, using the vocabulary and added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]

            return self._convert_id_to_token(ids)

        # current_stream will collect multiple tokens, and then separately add items
        # for each added token. This is done so that decode works properly with token ids
        # that cannot be represented naively in utf-8.
        tokens = []
        current_stream = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue

            if index in self.added_tokens_decoder:
                tokens.append(self.encoding.decode(current_stream))
                current_stream = []
                tokens.append(self.added_tokens_decoder[index])
            else:
                current_stream.append(index)

        if len(current_stream) > 0:
            tokens.append(self.encoding.decode(current_stream))
        return tokens

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
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

        Function copied from
        https://github.com/huggingface/transformers/blob/e3a4bd2bee212a2d0fd9f03b27fe7bfc1debe42d/src/transformers/models/gpt2/tokenization_gpt2.py#L265-L295

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
        # There is some code in huggingface that calls this function to get the vocab files,
        # but it doesn't seem to access them (or at least checks for their existence
        # before accessing them)
        return (None, None)  # type: ignore

    def sanitize_special_tokens(self) -> int:
        """Make sure that all the special tokens attributes of the tokenizer.

        (`tokenizer.mask_token`, `tokenizer.cls_token`, etc.) are in the
        vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            `int`: The number of tokens added in the vocabulary during the operation.
        """
        actual_new_tokens = []
        for token in self.all_special_tokens_extended:
            encoded = self.encoding.encode(token, allowed_special='all')
            if len(encoded) > 1:
                actual_new_tokens.append(token)

        return self.add_tokens(actual_new_tokens, special_tokens=True)

    def construct_logit_tensor(self, logprobs: Dict[str,
                                                    float]) -> torch.Tensor:
        """Construct tensor of shape (vocab_size,) mapping words to logprobs.

        Args:
            logprobs (Dict[str, float]): Dictionary mapping tokens to log probabilities assigned to them by the model.
        """
        tensor = torch.tensor([min(logprobs.values()) - 1] * (self.vocab_size))
        for k in logprobs:
            encoding = self(k)['input_ids']
            idx = encoding[0]
            tensor[idx] = logprobs[k]
        return tensor


TiktokenTokenizerWrapper.register_for_auto_class()
