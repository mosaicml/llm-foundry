import transformers
from peft import get_peft_model, LoraConfig
from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp

def test_peft_wraps():
    mistral_cfg = transformers.AutoConfig.from_pretrained('mistralai/Mistral-7B-v0.1', num_hidden_layers=2)
    mistral = transformers.AutoModelForCausalLM.from_config(mistral_cfg)
    mistral = get_peft_model(mistral, LoraConfig())
    prepare_hf_model_for_fsdp(mistral, 'cpu')
    assert False