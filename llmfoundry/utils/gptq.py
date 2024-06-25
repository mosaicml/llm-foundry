from transformers import AutoTokenizer
from datasets import load_dataset
import omegaconf
from pathlib import Path
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from composer.utils import reproducibility
import os
from auto_gptq import AutoGPTQForCausalLM
import transformers


def vmware_open_instruct_prepreprocess(example):
    return {"messages": [
        {
            'role': 'user', 
            'content': example['instruction']
        },
        {
            "role": "assistant",
            'content': example['response']
        }
    ]}

PREPREPROCESS_MAP = {
    "VMware/open-instruct": vmware_open_instruct_prepreprocess
}

def get_calibration_data(cfg: omegaconf.DictConfig):
    MAX_SEQ_LEN = cfg.get("max_seq_len", 8192)
    NUM_EXAMPLES = cfg.get("num_examples", 512)
    MODEL_ID = cfg.get("model_id")
    DATASET = cfg.get("dataset", "HuggingFaceH4/ultrachat_200k")
    SPLIT = cfg.get("split", "train_sft")
    
    prepreprocess = PREPREPROCESS_MAP.get(DATASET, lambda *_, **__: None)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = load_dataset(DATASET, split=SPLIT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = dataset.shuffle().select(range(NUM_EXAMPLES))
    ds = ds.map(prepreprocess)
    ds = ds.map(preprocess)

    examples = [
        tokenizer(
            example["text"], padding=False, max_length=MAX_SEQ_LEN, truncation=True,
        ) for example in ds
    ]
    return examples

# %%
def get_custom_calibration_data(cfg: omegaconf.DictConfig):
    MAX_SEQ_LEN = cfg.get("max_seq_len", 8192)
    MODEL_ID = cfg.get("model_id")
    DATASET_PATH = cfg.dataset
    CHUNK_SIZE = cfg.get("chunk_size", 512)
    
    dataset = load_dataset("text", data_files=DATASET_PATH, sample_by='document',split="train", download_mode="force_redownload")
    def chunk_examples(examples):
        chunks = []
        for sentence in examples["text"]:
            chunks += [sentence[i:i + CHUNK_SIZE] for i in range(0, len(sentence), CHUNK_SIZE)]
        return {"chunks": chunks}
    dataset = dataset.map(chunk_examples, batched=True, remove_columns=["text"])
    NUM_EXAMPLES = min(cfg.get("num_examples", 512), len(dataset))
    dataset.shuffle().select(range(NUM_EXAMPLES))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    examples = [
        tokenizer(
            example["chunks"], padding=False, max_length=MAX_SEQ_LEN, truncation=True,
        ) for example in dataset
    ]
    return examples

# %%
def quantize(cfg: omegaconf.DictConfig, examples, **kwargs):
    quantize_config = BaseQuantizeConfig(
        bits=cfg.wbits,
        group_size=cfg.gs,
        desc_act=cfg.actorder,
        clip=cfg.get('clip', False),
        mse=cfg.get('mse', False),
        sym=cfg.get('sym', True),
        static_groups=cfg.get('static_groups', False),
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        cfg.model_id,
        quantize_config,
        **kwargs)
    model.quantize(examples)

    return model


def should_quantize(cfg: omegaconf.DictConfig):
    if cfg.get('no_save', False):
        print('no_save flag is set, not loading or saving the newly quantized model')
        return True, None
    
    gptq_save_dir = cfg.get('gptq_save_dir')
    if gptq_save_dir:
        if Path(gptq_save_dir).is_dir():
            print(f"found gptq model at {gptq_save_dir}")
            return False, Path(gptq_save_dir)
        else:
            raise FileNotFoundError(f'{gptq_save_dir} not found, please provide a valid directory')
    else:
        try:
            gptq_save_dir = Path.home() / "saved" / \
                f"{cfg.model_id.split('/')[-1]}-gptq{cfg.wbits}-{cfg.gs}-{cfg.actorder}"
            if cfg.get('suffix'):
                gptq_save_dir = gptq_save_dir.with_name(
                    gptq_save_dir.name + f"-{cfg.get('suffix')}")
        except:
            raise ValueError("not enough information to determine if quantization is needed")
        if Path(gptq_save_dir).is_dir():
            print(f"found gptq model at the derived {gptq_save_dir}")
            return False, Path(gptq_save_dir)
        else:
            print(f"no gptq model found at the derived {gptq_save_dir}")
            return True, Path(gptq_save_dir)
            

def get_llama_marlin_factory(cfg, original_from_pretrained=None):
    def get_llama_marlin(*args, **kwargs):
        if original_from_pretrained:
            transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
        
        assert not (cfg.get('use_marlin', False) and cfg.get('no_save', False)), "no_save and use_marlin cannot be used together"
        
        _should_quantize, potential_save_dir = should_quantize(cfg)
        
        if _should_quantize:
            print('creating gptq model from following config:\n', cfg)
            if cfg.get('chunked', False):
                calibration_data = get_custom_calibration_data(cfg)
            else:
                calibration_data = get_calibration_data(cfg)
            model = quantize(cfg, calibration_data, **kwargs)

            if cfg.get('no_save', False):
                return model
            
            print(f"saving gptq model to {potential_save_dir}")
            model.save_quantized(potential_save_dir)
            del model
            import gc
            gc.collect()

        print(f"Loading gptq model from {potential_save_dir}")
        model = AutoGPTQForCausalLM.from_quantized(
            potential_save_dir, use_marlin=cfg.get('use_marlin', False), **kwargs)

        return model
    return get_llama_marlin


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    from omegaconf import OmegaConf
    cfg = OmegaConf.from_cli()
    cfg_default = OmegaConf.create(
        """
model_id: meta-llama/Meta-Llama-3-8B-Instruct
wbits: 4
gs: 128
actorder: True
no_save: True
clip: True
mse: 1
seed: 1
# suffix: seed${seed}-hqq
# sym: False
# dataset: VMware/open-instruct
# split: train
#   dataset: /nfs/scistore19/alistgrp/amoeini/group_10_merged.txt
#   chunked: true
#   chunk_size: 1024   
python_log_level: INFO   
        """
    )
    cfg = OmegaConf.merge(cfg_default, cfg)
    if cfg.get('seed'):
        reproducibility.seed_all(cfg.seed)
    import logging
    logging.basicConfig(level=cfg.get('python_log_level', 'WARNING'))
    model = get_llama_marlin_factory(cfg)()
    
# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, GPTQConfig
#     gptq_config = GPTQConfig(bits=4, use_exllama=False)
#     model1 = AutoModelForCausalLM.from_pretrained("/nfs/scistore19/alistgrp/amoeini/saved/Meta-Llama-3-8B-Instruct-gptq4-128-True-seed1-clip", device_map="auto", quantization_config=gptq_config)
#     pass