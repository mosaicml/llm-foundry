# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import torch
from InstructorEmbedding import INSTRUCTOR


class BaseModelHandler:
    """Base class that all custom model handlers should inherit from."""

    def __init__(self) -> None:
        self.device = torch.cuda.current_device()
        self.setup()
        print(f'DSInferenceModel initialized with device: {self.device}')

    def setup(self):
        raise NotImplementedError

    def predict(self, **inputs: Dict[str, Any]):
        raise NotImplementedError


class HFInstructorLargeModel(BaseModelHandler):
    """Custom Hugging Face Instructor Model handler class."""
    MODEL_NAME = 'hkunlp/instructor-large'
    model: INSTRUCTOR = None

    def setup(self):
        """Loads and instantiates the Hugging Face Large Instructor model."""
        self.model = INSTRUCTOR(self.MODEL_NAME)

    def predict(self, **inputs: Dict[str, Any]):
        """Runs a forward pass with the given inputs.

        Inputs must be dictionary that contains a 'input_strings' key with the
        value being in the following format:     [<instruction>, <sentence>]
        """
        # input_strings in format of [[<instruction>, <sentence>]]
        if 'input_strings' not in inputs:
            raise KeyError('input_strings key not in inputs')

        embeddings = self.model.encode(inputs['input_strings'])
        return embeddings.tolist()
