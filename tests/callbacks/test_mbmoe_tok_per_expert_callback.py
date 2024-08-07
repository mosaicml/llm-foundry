# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.utils.builders import build_callback


def test_mbmoe_tok_per_expert_builds():
    """Test that the callback can be built."""
    callback = build_callback(name='mbmoe_tok_per_expert')
    assert callback is not None
    assert callback.__class__.__name__ == 'MegaBlocksMoE_TokPerExpert'
