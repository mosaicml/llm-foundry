from transformers.models.llama.modeling_llama import LlamaAttention
import transformers
import torch



def test_patch_equivalence():
    llama_7b_config = transformers.AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=True)
    original_attention = LlamaAttention(
        config=llama_7b_config,
    )

    hidden_states = torch.randn(2, 4096, 4096)
    attn_output, attn_weights, past_key_value = original_attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    )

    from llmfoundry.models.layers.llama_attention_monkeypatch import new_forward

    new_output, new_weights, new_past_key_value = original_attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    )

    assert False