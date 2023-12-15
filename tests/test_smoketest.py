# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry import callbacks, data, models, optim, tokenizers, utils


# This very simple test is just to use the above imports, which check and make sure we can import all the top-level
# modules from foundry. This is mainly useful for checking that we have correctly conditionally imported all optional
# dependencies.
def test_smoketest():
    assert callbacks
    assert data
    assert models
    assert optim
    assert tokenizers
    assert utils
