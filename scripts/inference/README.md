# LLM Inference

This folder contains helper scripts for exporting and generating outputs with your Composer-trained LLMs.

Table of Contents:
- [Converting a Composer checkpoint to an HF checkpoint folder](#converting-a-composer-checkpoint-to-an-hf-checkpoint-folder)
- [Interactive Generation with HF models](#interactive-generation-with-hf-models)
- [Interactive Chat with HF models](#interactive-chat-with-hf-models)
- [Converting an HF model to ONNX](#converting-an-hf-model-to-onnx)
- [Converting an HF MPT to FasterTransformer](#converting-an-hf-mpt-to-fastertransformer)

## Converting a Composer checkpoint to an HF checkpoint folder

The LLMs trained with this codebase are all HuggingFace (HF) `PreTrainedModel`s, which we wrap with a `HuggingFaceModel` wrapper class to make compatible with Composer. See [docs](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.models.HuggingFaceModel.html#huggingfacemodel) and an [example](https://docs.mosaicml.com/projects/composer/en/latest/examples/pretrain_finetune_huggingface.html) for more details.

At the end of your training runs, you will see a collection of Composer `Trainer` checkpoints such as `ep0-ba2000-rank0.pt`. These checkpoints contain the entire training state, including the model, tokenizer, optimizer state, schedulers, timestamp, metrics, etc. Though these Composer checkpoints are useful during training, at inference time we usually just want the model, tokenizer, and metadata.

To extract these pieces, we provide a script `convert_composer_to_hf.py` that converts a Composer checkpoint directly to a standard HF checkpoint folder. For example:

<!--pytest.mark.skip-->
```bash
python convert_composer_to_hf.py --composer_path ep0-ba2000-rank0.pt --hf_output_path my_hf_model/ --output_precision bf16
```

This will produce a folder like:

```
my_hf_model/
  config.json
  merges.txt
  pytorch_model.bin
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  modeling_code.py
```

which can be loaded with standard HF utilities like `AutoModelForCausalLM.from_pretrained('my_hf_model')`.

You can also pass object store URIs for both `--composer_path` and `--hf_output_path` to easily convert checkpoints stored in S3, OCI, etc.

## Interactive Generation with HF models

To make it easy to inspect the generations produced by your HF model, we include a script `hf_generate.py` that allows you to run custom prompts through your HF model, like so:

<!--pytest.mark.skip-->
```bash
python hf_generate.py \
    --name_or_path gpt2 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 256 \
    --prompts \
      "The answer to life, the universe, and happiness is" \
      "MosaicML is an ML training efficiency startup that is known for" \
      "Here's a quick recipe for baking chocolate chip cookies: Start by" \
      "The best 5 cities to visit in Europe are"
```

which will produce output:

<!--pytest.mark.skip-->
```bash
Loading HF model...
n_params=124439808

Loading HF tokenizer...
/mnt/workdisk/llm-foundry/scripts/inference/hf_generate.py:89: UserWarning: pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.
  warnings.warn(
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

Generate kwargs:
{'max_new_tokens': 256, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 50, 'use_cache': True, 'do_sample': True, 'eos_token_id': 50256}

Moving model and inputs to device=cuda and dtype=torch.bfloat16...

Tokenizing prompts...
NOT using autocast...
Warming up...
Generating responses...
####################################################################################################
The answer to life, the universe, and happiness is to love...
####################################################################################################
MosaicML is an ML training efficiency startup that is known for designing and developing applications to improve training and performance efficiency...
####################################################################################################
Here\'s a quick recipe for baking chocolate chip cookies: Start by making an apple crumble by yourself or bake in the microwave for 40 minutes to melt and get melted...
####################################################################################################
The best 5 cities to visit in Europe are the one in Spain (Spain) and the one in Holland (Belgium)...
####################################################################################################
bs=4, input_tokens=array([11, 14, 13,  9]), output_tokens=array([256, 256, 256,  41])
total_input_tokens=47, total_output_tokens=809
encode_latency=9.56ms, gen_latency=2759.02ms, decode_latency=1.72ms, total_latency=2770.31ms
latency_per_output_token=3.42ms/tok
output_tok_per_sec=292.03tok/sec
```

The argument for `--name_or_path` can be either the name of a model that exists on the HF Hub, such as `gpt2`, `facebook/opt-350m`, etc. or the path to a HF checkpoint folder, such as `my_hf_model/` like we exported above.

## Interactive Chat with HF models

Chat models need to pass conversation history back to the model for multi-turn conversations. To make that easier, we include `hf_chat.py`. Chat models usually require an introductory/system prompt, as well as a wrapper around user and model messages, to fit the training format. Default values work with our ChatML-trained models, but you can specify these values with CLI args:

<!--pytest.mark.skip-->
```bash
python hf_chat.py -n my_hf_model/ --system_prompt="You are a helpful assistant\n" --user_msg_fmt="user: {}\n" --assistant_msg_fmt="assistant: {}\n" --max_new_tokens=512
```

## Converting an HF model to ONNX

We include a script `convert_hf_to_onnx.py` that demonstrates how to convert your HF model to ONNX format. For more details and examples
of exporting and working with HuggingFace models with ONNX, see <https://huggingface.co/docs/transformers/serialization#export-to-onnx>.

Here a couple examples of using the script:

<!--pytest.mark.skip-->
```bash
# 1) Local export
python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder local/folder

# 2) Remote export
python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder s3://bucket/remote/folder

# 3) Verify the exported model
python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder local/folder --verify_export

# 4) Change the batch size or max sequence length
python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder local/folder --export_batch_size 1 --max_seq_len 32000
```

Please open a Github issue if you discover any problems!

## Converting an HF MPT to FasterTransformer
We include a script `convert_hf_mpt_to_ft.py` that converts HuggingFace MPT checkpoints to the
[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) format. This makes the checkpoints compatible with the
FasterTransformer library, which can be used to run transformer models on GPUs.

You can either pre-download the model in a local dir or directly provide the HF hub model name to convert the HF MPT
checkpoint to FasterTransformer format.

### Download and Convert
```
# The script handles the download
python convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o mpt-ft-7b --infer_gpu_num 1
```

### Pre-Download the Model and Convert
```
apt update
apt install git-lfs
git lfs install
git clone https://huggingface.co/mosaicml/mpt-7b
# This will convert the MPT checkpoint in mpt-7b dir and save the converted checkpoint to mpt-ft-7b dir
python convert_hf_mpt_to_ft.py -i mpt-7b -o mpt-ft-7b --infer_gpu_num 1
```
You can change `infer_gpu_num` to > 1 to prepare a FT checkpoint for multi-gpu inference. Please open a Github issue if you discover any problems!
