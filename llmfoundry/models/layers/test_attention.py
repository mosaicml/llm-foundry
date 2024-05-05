import unittest
import torch
from llmfoundry.models.layers.attention import StateSpaceAttention
from llmfoundry.models.layers.mamba_layer import MambaLayer

class TestAttentionLayers(unittest.TestCase):

    def test_state_space_attention(self):
        # Initialize parameters for the StateSpaceAttention
        d_model = 16  # Corrected d_model to match the expected dimension
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
        self.assertEqual(context.shape, (batch_size, seq_length, d_model), "Context output shape is incorrect")

        # Check the shape of the attention probabilities
        self.assertEqual(attn_probs.shape, (batch_size, n_heads, seq_length, seq_length), "Attention probabilities shape is incorrect")

        # Check that the attention probabilities sum to 1 across the last dimension
        attn_probs_sum = attn_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(attn_probs_sum, torch.ones_like(attn_probs_sum)), "Attention probabilities do not sum to 1")

    def test_mamba_layer(self):
        # Initialize parameters for the MambaLayer
        d_model = 16  # Corrected d_model to match the expected dimension
        ssm_params = {
            'A': torch.randn(d_model, d_model),
            'B': torch.randn(d_model, d_model),
            'C': torch.randn(d_model, d_model),
            'D': torch.randn(d_model, d_model)
        }
        batch_size = 2
        seq_length = 10

        # Instantiate the MambaLayer class
        mamba_layer = MambaLayer(d_model, ssm_params)

        # Create mock data for input
        x = torch.rand(batch_size, seq_length, d_model)

        # Forward pass through the MambaLayer
        y = mamba_layer(x)

        # Check the shape of the output
        self.assertEqual(y.shape, (batch_size, seq_length, d_model), "Output shape is incorrect")

if __name__ == '__main__':
    unittest.main()
