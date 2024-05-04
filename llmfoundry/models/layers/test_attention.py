import torch
import torch.nn as nn
from llmfoundry.models.layers.attention import StateSpaceAttention

def test_state_space_attention():
    # Initialize parameters for the StateSpaceAttention
    d_model = 64
    n_heads = 8
    state_space_size = 16
    batch_size = 2
    seq_length = 10

    # Instantiate the StateSpaceAttention class
    ssa = StateSpaceAttention(d_model, n_heads, state_space_size)

    # Create mock data for query, key, and value
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)

    # Forward pass through the StateSpaceAttention layer
    context, attn_probs = ssa(query, key, value)

    # Check the shape of the output context
    assert context.shape == (batch_size, seq_length, d_model), "Context output shape is incorrect"

    # Check the shape of the attention probabilities
    assert attn_probs.shape == (batch_size, n_heads, seq_length, seq_length), "Attention probabilities shape is incorrect"

    # Check that the attention probabilities sum to 1 across the last dimension
    attn_probs_sum = attn_probs.sum(dim=-1)
    assert torch.allclose(attn_probs_sum, torch.ones_like(attn_probs_sum)), "Attention probabilities do not sum to 1"

    print("All tests passed for StateSpaceAttention.")

if __name__ == "__main__":
    test_state_space_attention()
