
import torch
import os
import numpy as np
import logging
from typing import Any, Optional, Dict, List, Tuple, Union

from chronos import chronos
from chronos import ChronosConfig, ChronosTokenizer, ChronosPipeline
from transformers import PreTrainedTokenizer, AutoConfig, AddedToken, BatchEncoding
from transformers.tokenization_utils_base import EncodedInput, EncodedInputPair, PreTokenizedInput, PreTokenizedInputPair, TextInput, TextInputPair
from transformers.utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy

__all__ = [
    'ChronosTokenizerWrapper',
]

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."""

logger = logging.getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# Using existing ChronosTokenizer and transformers.PreTrainedTokenizer interface
class ChronosTokenizerWrapper(PreTrainedTokenizer):
    
    chronos_tokenizer: ChronosTokenizer
    pad_token_id: int
    eos_token_id: int
    
    def __init__(self, tokenizer_name: Optional[str]='amazon/chronos-t5-small', **kwargs: Dict[str, Any]):
        config = AutoConfig.from_pretrained(
            tokenizer_name,
            **kwargs,
        )
        # device_map = kwargs.get('device_map')  # used for ChronosPipeline
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        chronos_config = ChronosConfig(**config.chronos_config)  # dictionary of content in "chronos_config" of amazon/chronos-t5-{} config.json
        self.chronos_tokenizer = chronos_config.create_tokenizer()  # type: chronos.chronos.MeanScaleUniformBins
        # Get the ChronosConfig object with: self.chronos_tokenizer.config
        
        # Initialize PreTrainedTokenizer attributes
        super().__init__(
            pad_token=AddedToken(str(self.chronos_tokenizer.config.pad_token_id)), 
            eos_token=AddedToken(str(self.chronos_tokenizer.config.eos_token_id)), 
            **kwargs
        )
        
        
    def get_vocab(self) -> Dict[float, float]:
        return {token: token_id for token, token_id in zip(self.chronos_tokenizer.centers.tolist(), list(range(self.chronos_tokenizer.config.n_special_tokens, self.chronos_tokenizer.config.n_tokens)))}
        # return {i.item(): i.item() for i in self.chronos_tokenizer.boundaries}

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.chronos_tokenizer.config.n_tokens
    
    # @property
    # def eos_token_id(self) -> Optional[int]:
    #     """
    #     `Optional[int]`: Id of the end of sentence token in the vocabulary.
    #     """
    #     return self.chronos_tokenizer.config.eos_token_id
    
    # @property
    # def pad_token_id(self) -> Optional[int]:
    #     """
    #     `Optional[int]`: Id of the padding token in the vocabulary.
    #     """
    #     return self.chronos_tokenizer.config.pad_token_id
    
    # def _prepare_and_validate_context(self, context: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    #     if isinstance(context, list):
    #         context = left_pad_and_stack_1D(context)
    #     assert isinstance(context, torch.Tensor)
    #     if context.ndim == 1:
    #         context = context.unsqueeze(0)
    #     assert context.ndim == 2

    #     return context
    
    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # print(f'save_directory == {save_directory}')
        # index = 0
        # if os.path.isdir(save_directory):
        #     vocab_file = os.path.join(
        #         save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        #     )
        # else:
        #     vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # with open(vocab_file, "w", encoding="utf-8") as writer:
        #     for token, token_index in sorted(self.get_vocab().items(), key=lambda kv: kv[1]):
        #     # for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
        #         if index != token_index:
        #             logger.warning(
        #                 f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
        #                 " Please check that the vocabulary is not corrupted!"
        #             )
        #             index = token_index
        #         writer.write(str(token) + "\n")
        #         index += 1
        # return (vocab_file,)
        
        return (save_directory,)
    
        
        # vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        # with open(save_directory, 'w', encoding='utf-8') as f:
        #     for token, index in sorted(self.get_vocab(), key=lambda item: item[1]):
        #         f.write(f"{token}\n")
        
        # return (save_directory,)

    # def convert_tokens_to_ids(self, tokens: Union[torch.Tensor, List[torch.Tensor]]) -> Union[int, List[int]]:
    #     """
    #     ORIGINAL: Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary.
    #     MODIFIED: Converts a token tensor (or a list of tokens) in a single integer id (or a sequence of ids), using context_input_transform
        
    #     Note: No need for _convert_token_to_id_with_added_voc or _convert_token_to_id helper methods
    #     """
    #     token_ids, _, tokenizer_state = self.chronos_tokenizer.context_input_transform(context=tokens)
    #     # TODO: implement how to return (or store) the `tokenizer_state` of the transformation
    #     return token_ids

    # def convert_ids_to_tokens(self, ids: torch.Tensor, scale: torch.Tensor, skip_special_tokens: bool = False) -> torch.Tensor:
    #     """
    #     Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and added tokens.
        
    #     Note: No need for _convert_id_to_token helper method
    #     """
    #     # TODO: Implement how to store `scale`
    #     return self.chronos_tokenizer.output_transform(samples=ids, scale=scale)
    
    # def encode(self, context: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Converts a string to a sequence of ids (integers), using the tokenizer and vocabulary.
    #     Same as doing `self.convert_tokens_to_ids(self.tokenize(text)).
    #     """
    #     context_tensor = self._prepare_and_validate_context(context=context)
    #     token_ids, attention_mask, tokenizer_state = self.chronos_tokenizer.context_input_transform(context=context_tensor)
    #     return token_ids, attention_mask, tokenizer_state  # return sequence of ids (integers), and the scale (tokenizer_state)
    #     # return token_ids
    #     # TODO: Need to figure out how to store `tokenizer_state` upon returning
        
    # def _encode_plus(self, context: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # TODO: Might need to re-implement some features
    #     return self.encode(context, *args, **kwargs)
    
    # def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]]) -> BatchEncoding:
    #     # TODO: Might need to re-implement
    #     return super()._batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs)
        
    # def _batch_prepare_for_model(self, batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]]) -> BatchEncoding:
    #     """
    #     Prepares a sequence of input id, or a pair of sequences of input ids so that it can be used by the model.
    #     It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and manages a moving window (with user defined stride) for overflowing tokens.
        
    #     Args:
    #         batch_ids_pairs: list of tokenized input ids or input ids pairs
    #     """
    #     # TODO: Might need to implement
    #     return super()._batch_prepare_for_model(batch_ids_pairs=batch_ids_pairs)

    # def decode(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:  # PreTrainedTokenizer.decode(token_ids,...) returns `str`
    #     # Assuming `scale` is passed in kwargs for decoding
    #     scale = kwargs.get('scale')
    #     if scale is None:
    #         raise ValueError("Scale must be provided for decoding.")
    #     return self.chronos_tokenizer.output_transform(token_ids, scale)
    
    # def _decode(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    #     # Do not need this helper method
    #     return self.decode(token_ids=token_ids, **kwargs)
    
    # def tokenize(self, tokens: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
    #     """
    #     ORIGINAL: Converts a string into a sequence of tokens, using the tokenizer. Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies. 
    #             Takes care of added tokens.
    #     MODIFIED: Tokens are already split into real-valued numbers in tensors. No need to perform additional tokenization.
    #     """
    #     return tokens
    
    # def _tokenize(self, tokens: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
    #     """
    #     ORIGINAL: Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.
    #             Do NOT take care of added tokens.
    #     MODIFIED: Tokens are already split into real-valued numbers in tensors. No need to perform additional tokenization.
        
    #     Note: This method is not needed since `tokenize` does not use it as a helper method.
    #     """
    #     return self.tokenize(tokens=tokens, **kwargs)
    
    
    # def _add_tokens(self, new_tokens: Union[torch.Tensor, List[torch.Tensor]], special_tokens: bool = False) -> int:
    #     """
    #     Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary they are added to it with indices starting from length of the current vocabulary.
    #     Special tokens are sometimes already in the vocab which is why they have to be handled specifically.
        
    #     Returns:
    #         `int`: The number of tokens actually added to the vocabulary.
    #     """
    #     # TODO: might need to implement this
    #     return super()._add_tokens(new_tokens=new_tokens, special_tokens=special_tokens)
    
    # def prepare_for_tokenization(self, tokens: Union[torch.Tensor, List[torch.Tensor]], is_split_into_words: bool = False, **kwargs) -> Tuple[torch.Tensor, Dict[torch.Tensor, any]]:
    #     """
    #     Performs any necessary transformations before tokenization.
    #     This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the `kwargs` at the end of the encoding process to be sure all the arguments have been used.
        
    #     Returns:
    #         `Tuple[torch.Tensor, Dict[torch.Tensor, Any]]`: The prepared tensor and the unused kwargs
            
    #     Note: Same implementation as PreTrainedTokenizer.prepare_for_tokenization for now
    #     """
    #     # TODO: Might need to add any data preparation
    #     return super().prepare_for_tokenization(text=tokens, is_split_into_words=is_split_into_words)
    
    
    def _to_hf_format(self, entry: Dict[str, Any]) -> Dict[str, Any]:     
        # Pad horizon to acceptable length - checked in label_input_transform()
        horizon = entry['future_target']
        padding = np.full(((self.chronos_tokenizer.config.prediction_length - len(horizon)),), np.nan, dtype=horizon.dtype)
        entry['future_target'] = np.concatenate([horizon, padding])
        
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.chronos_tokenizer.context_input_transform(context=past_target)
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.chronos_tokenizer.label_input_transform(label=future_target, scale=scale)
        labels[labels_mask == 0] = -100
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }
    
    # This method will get called is there is a self.tokenizer(...) call in another class, where self.tokenizer has type `ChronosTokenizerWrapper`
    # def __call__(
    #     self,
    #     tokens: Union[torch.Tensor, np.ndarray] = None,
    #     add_special_tokens: bool = True,
    #     padding: Union[bool, str, PaddingStrategy] = False,
    #     truncation: Union[bool, str, TruncationStrategy] = None,
    #     max_length: Optional[int] = None,
    #     stride: int = 0,
    #     is_split_into_words: bool = False,
    #     pad_to_multiple_of: Optional[int] = None,
    #     return_tensors: Optional[Union[str, TensorType]] = None,
    #     return_token_type_ids: Optional[bool] = None,
    #     return_attention_mask: Optional[bool] = None,
    #     return_overflowing_tokens: bool = False,
    #     return_special_tokens_mask: bool = False,
    #     return_offsets_mapping: bool = False,
    #     return_length: bool = False,
    #     verbose: bool = True,
    #     **kwargs: Dict[str, Any],
    # ) -> BatchEncoding:
    #     """
    #     Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
    #     sequences.

    #     Args:
    #         tokens (`torch.Tensor`, *optional*):
    #             The sequence to be encoded. Each sequence must be a torch.Tensor. 
    #             If the sequences are provided as list of tensors (pretokenized), you must set
    #             `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    #     """
    #     print(f'tokens (inside __call__) == {tokens}')
    #     # To avoid duplicating
    #     all_kwargs = {
    #         "add_special_tokens": add_special_tokens,
    #         "padding": padding,
    #         "truncation": truncation,
    #         "max_length": max_length,
    #         "stride": stride,
    #         "is_split_into_words": is_split_into_words,
    #         "pad_to_multiple_of": pad_to_multiple_of,
    #         "return_tensors": return_tensors,
    #         "return_token_type_ids": return_token_type_ids,
    #         "return_attention_mask": return_attention_mask,
    #         "return_overflowing_tokens": return_overflowing_tokens,
    #         "return_special_tokens_mask": return_special_tokens_mask,
    #         "return_offsets_mapping": return_offsets_mapping,
    #         "return_length": return_length,
    #         "verbose": verbose,
    #     }
    #     all_kwargs.update(kwargs)
    #     if tokens is not None:
    #         # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
    #         # input mode in this case.
    #         if not self._in_target_context_manager:
    #             self._switch_to_input_mode()
    #         # encodings = self._call_one(text=tokens, text_pair=text_pair, **all_kwargs)
    #         if isinstance(tokens, np.ndarray):
    #             tokens = torch.from_numpy(tokens)
    #         print(f'type(tokens) :: {type(tokens)}')
    #         print(f'tokens.shape :: {tokens.shape}')
    #         # token_ids, attention_mask, tokenizer_state = self.chronos_tokenizer.context_input_transform(context=tokens)
    #     # Leave back tokenizer in input mode
    #     self._switch_to_input_mode()
    #     encodings = self._to_hf_format(tokens)
    #     print(f'encodings (inside __call__) == {encodings}')
    #     return BatchEncoding(data=encodings)



def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor: 
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)

ChronosTokenizerWrapper.register_for_auto_class()
