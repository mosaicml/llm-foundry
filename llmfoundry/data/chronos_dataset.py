
# import ast
import logging
# import os
# import re
# import sys
# import json
import itertools
# import random
# from copy import deepcopy
# from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

# import typer
# from typer_config import use_yaml_config
import numpy as np
import torch
# import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
# import transformers
# from transformers import (
#     AutoModelForSeq2SeqLM,
#     AutoModelForCausalLM,
#     AutoConfig,
#     T5Config,
#     Trainer,
#     TrainingArguments,
# )
# import accelerate
import gluonts
# from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
)

from chronos import ChronosConfig, ChronosTokenizer



def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob
        self.min_past = min_past or prediction_length
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        logging.debug(f'chronos_dataset.ChronosDataset.create_training_data(): data :: {data}')
        logging.debug(f'chronos_dataset.ChronosDataset.create_training_data(): type(data) :: {type(data)}')
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        
        
        labels[labels_mask == 0] = -100
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        logging.debug('chronos_dataset.ChronosDataset.__iter__(): About to create `iterables`...')
        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]
        logging.debug(f'chronos_dataset.ChronosDataset.__iter__(): chronos_dataset.ChronosDataset.__iter__() :: Created `iterables` with mode {self.mode}')

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )
        logging.debug(f'chronos_dataset.ChronosDataset.__iter__(): Iterables :: {iterables}')
        logging.debug(f'chronos_dataset.ChronosDataset.__iter__(): Probabilities before normalization :: {probs}')
        
        # if not iterables:  # I added this check
        #     logging.debug(f"chronos_dataset.ChronosDataset.__iter__(): When iterables is NONE, self.mode == {self.mode}, self.datasets == {self.datasets}, preprocessed_datasets == {preprocessed_datasets}")
        #     raise ValueError("Iterables are empty. Please check the dataset initialization.")

        probs = [prob / sum(probs) for prob in probs]
        logging.debug(f'chronos_dataset.ChronosDataset.__iter__(): Probabilities after normalization: {probs}')

        iterators = list(map(iter, iterables))
        # if not iterators:  # I added this check
        #     raise ValueError("Iterators are empty. Please check the dataset initialization.")
        logging.debug(f'chronos_dataset.ChronosDataset.__iter__(): iterators :: {iterators}')
        logging.debug(f'chronos_dataset.ChronosDataset.__iter__(): len(iterators) :: {len(iterators)}')  # 1 iterator per dataset passed in
        
        if self.mode == "training":
            i = 1
            # Seems to run for 100,126 iterations under standard configurations, stops (50%), then continued until 100,158 iterations (100%) (depends on `shuffle_buffer_length`)
            while True:
                # `idx` = 0 for a single dataset
                # logging.debug(f'i == {i}, iterators == {iterators}, len(iterators) == {len(iterators)}, probs == {probs}, range(len(iterators)) == {range(len(iterators))}')
                try:
                    idx = np.random.choice(range(len(iterators)), p=probs)
                except:
                    # print(f'ERROR (i == {i}): {e}')
                    return
                try:
                    # value = next(iterators[idx])
                    # value_hf = self.to_hf_format(value)
                    # if i == 1:
                    #     logging.info(f'chronos_dataset.ChronosDataset.__iter__() > while: Example `next(iterators[idx])` looks like :: {value}')
                    #     logging.info(f'chronos_dataset.ChronosDataset.__iter__() > while: Example `self.to_hf_format(next(iterators[idx]))` looks like :: {value_hf}')
                    # if list(value.keys()) != ['start', 'past_target', 'future_target', 'past_is_pad', 'forecast_start']:
                    #     logging.debug(f'chronos_dataset.ChronosDataset.__iter__() > while: Incorrect value at idx={idx} :: {value}')
                    # if list(value_hf.keys()) != ['input_ids', 'attention_mask', 'labels']:
                    #     logging.debug(f'chronos_dataset.ChronosDataset.__iter__() > while: Incorrect value_hf at idx={idx} :: {value_hf}')
                    # yield value_hf
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    # return
                    logging.debug(f'chronos_dataset.ChronosDataset.__iter__() > while: Stopping iteration when i = {i}')  # Does not reach this
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
                # print(f'i == {i}, idx == {idx}')
                i += 1
        else:
            logging.debug('chronos_dataset.ChronosDataset.__iter__(): Entering the case when self.mode != training')  # Does not reach this
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)
