from transformers import AutoTokenizer
from datasets import load_dataset
import omegaconf
from pathlib import Path
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from composer.utils import reproducibility
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def get_calibration_data(cfg: omegaconf.DictConfig):
    MAX_SEQ_LEN = cfg.get("max_seq_len", 8192)
    NUM_EXAMPLES = cfg.get("num_examples", 512)
    MODEL_ID = cfg.get("model_id")
    DATASET = cfg.get("dataset", "HuggingFaceH4/ultrachat_200k")

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = load_dataset(DATASET, split="train_sft")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = dataset.shuffle().select(range(NUM_EXAMPLES))
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
def quantize_and_save(cfg: omegaconf.DictConfig, examples, path, **kwargs):
    quantize_config = BaseQuantizeConfig(
        bits=cfg.wbits,
        group_size=cfg.gs,
        desc_act=cfg.actorder,
        clip=cfg.get('clip', False),
        mse=cfg.get('mse', False),
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        cfg.model_id,
        quantize_config,
        **kwargs)
    model.quantize(examples)

    if cfg.get('no_save', False):
        return model
    print(f"Saving gptq model to {path}")
    model.save_quantized(path)

    import gc
    del model
    gc.collect()
    return None


def get_llama_marlin_factory(cfg, original_from_pretrained=None):
    def get_llama_marlin(*args, **kwargs):
        if original_from_pretrained:
            import transformers
            transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
        from auto_gptq import AutoGPTQForCausalLM

        gptq_save_dir = cfg.get('gptq_save_dir')
        if gptq_save_dir:
            gptq_save_dir = Path(gptq_save_dir)
        else:
            gptq_save_dir = Path.home() / "saved" / \
                f"{cfg.model_id.split('/')[-1]}-gptq{cfg.wbits}-{cfg.gs}-{cfg.actorder}"
            if cfg.get('suffix'):
                gptq_save_dir = gptq_save_dir.with_name(
                    gptq_save_dir.name + f"-{cfg.get('suffix')}")

        # check if checkpoint exists as a folder
        if not gptq_save_dir.is_dir() or cfg.get('no_save', False):
            if cfg.get('no_save', False):
                assert not cfg.get('use_marlin', False), "use_marlin=True is not supported with no_save=True"
            print(f'{gptq_save_dir} not found, creating...')
            print(cfg)
            if cfg.get('chunked', False):
                calibration_data = get_custom_calibration_data(cfg)
            else:
                calibration_data = get_calibration_data(cfg)
            model = quantize_and_save(cfg, calibration_data, gptq_save_dir, **kwargs)
            if cfg.get('no_save', False):
                return model

        print(f"Loading gptq model from {gptq_save_dir}")
        model = AutoGPTQForCausalLM.from_quantized(
            gptq_save_dir, use_marlin=cfg.get('use_marlin', False), **kwargs)

        return model
    return get_llama_marlin


if __name__ == "!__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.from_cli()
    cfg_default = OmegaConf.create(
        """
      model_id: meta-llama/Meta-Llama-3-8B-Instruct
      wbits: 4
      gs: 128
      actorder: True
    #   no_save: True
    #   clip: True
      seed: 1
      suffix: seed${seed}-clip
    #   dataset: /nfs/scistore19/alistgrp/amoeini/group_10_merged.txt
    #   chunked: true
    #   chunk_size: 1024      
        """
    )
    cfg = OmegaConf.merge(cfg_default, cfg)
    if cfg.get('seed'):
        cfg.suffix = f"seed{cfg.seed}"
        reproducibility.seed_all(cfg.seed)
    import os 
    if os.environ.get("DEBUG"):
        import logging
        logging.basicConfig(level=logging.DEBUG)
    model = get_llama_marlin_factory(cfg)()
    
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, GPTQConfig
    gptq_config = GPTQConfig(bits=4, use_exllama=False)
    model1 = AutoModelForCausalLM.from_pretrained("ISTA-DASLab/Llama-3-8B-Instruct-GPTQ-4bit", device_map="auto", quantization_config=gptq_config)
    pass