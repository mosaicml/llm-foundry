# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.callbacks.hf_checkpointer import HuggingFaceCheckpointer

# uses the oras client to save the model to the harbor registry


class HarborCheckpointer(HuggingFaceCheckpointer):
    pass
