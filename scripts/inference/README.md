# LLM Inference

This folder contains helper scripts for exporting and generating outputs with your Composer-trained LLMs.

Table of Contents:

- [LLM Inference](#llm-inference)
  - [Converting a Composer checkpoint to an HF checkpoint folder](#converting-a-composer-checkpoint-to-an-hf-checkpoint-folder)
  - [Interactive Generation with HF models](#interactive-generation-with-hf-models)
  - [Interactive Chat with HF models](#interactive-chat-with-hf-models)
  - [Converting an HF model to ONNX](#converting-an-hf-model-to-onnx)
  - [Converting an HF MPT to FasterTransformer](#converting-an-hf-mpt-to-fastertransformer)
    - [Download and Convert](#download-and-convert)
    - [Pre-Download the Model and Convert](#pre-download-the-model-and-convert)
  - [Converting a Composer MPT to FasterTransformer](#converting-a-composer-mpt-to-fastertransformer)
  - [Running MPT with FasterTransformer](#running-mpt-with-fastertransformer)
  - [Running MPT with TensorRT-LLM](#running-mpt-with-tensorrt-llm)

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

The script will use HuggingFace's `device_map=auto` feature to automatically load the model on any available GPUs, or fallback to CPU. [See the docs here!](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
You can also directly specify `--device_map auto` or `--device_map balanced`, etc.
You can also target a specific **single** device using `--device cuda:0` or `--device cpu`, etc.

For MPT models specifically, you can pass args like `--attn_impl flash`, and `--max_seq_len 4096` to speed up generation time or alter the max generation length at inference time (thanks to ALiBi).

## Interactive Chat with HF models

Chat models need to pass conversation history back to the model for multi-turn conversations. To make that easier, we include `hf_chat.py`. Chat models usually require an introductory/system prompt, as well as a wrapper around user and model messages, to fit the training format. Default values work with our ChatML-trained models, but you can set other important values like generation kwargs:

<!--pytest.mark.skip-->

```bash
# using an MPT/ChatML style model
python hf_chat.py -n mosaicml/mpt-7b-chat-v2 \
  --max_new_tokens=2048 \
  --temperature 0.3 \
  --top_k 0 \
  --model_dtype bf16 \
  --trust_remote_code
```

<!--pytest.mark.skip-->

```bash
# using an MPT/ChatML style model on  > 1 GPU
python hf_chat.py -n mosaicml/mpt-7b-chat-v2 \
  --max_new_tokens=1024 \
  --temperature 0.3 \
  --top_k 0 \
  --model_dtype bf16 \
  --trust_remote_code \
  --device_map auto
```

The script also works with other style models. Here is an example of using it with a Vicuna-style model:

<!--pytest.mark.skip-->

```bash
python hf_chat.py -n eachadea/vicuna-7b-1.1 --system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions." --user_msg_fmt="USER: {}\n" --assistant_msg_fmt="ASSISTANT: {}\n" --max_new_tokens=512
```

The `system_prompt` is the message that gives the bot context for the conversation, and can be used to make the bot take on different personalities.

In the REPL you see while using `hf_chat.py` you can enter text to interact with the model (hit return TWICE to send, this allows you to input text with single newlines), you can also enter the following commands:

- `clear` — clear the conversation history, and start a new conversation (does not change system prompt)
- `system` — change the system prompt
- `history` — see the conversation history
- `quit` — exit

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

## Converting a Composer MPT to FasterTransformer

We include a script `convert_composer_mpt_to_ft.py` that directly converts a Composer MPT checkpoint to the FasterTransformer format. You can either provide a path to a local Composer checkpoint or a URI to a file stored in a cloud supported by Composer (e.g. `s3://`). Simply run:

```
python convert_composer_mpt_to_ft.py -i <path_to_composer_checkpoint.pt> -o mpt-ft-7b --infer_gpu_num 1
```

## Running MPT with FasterTransformer

This step assumes that you already have converted an MPT checkpoint to FT format by following the instructions in
[Converting an HF MPT to FasterTransformer](#converting-an-hf-mpt-to-fastertransformer). It also assumes that you have

1. Built FasterTransformer for PyTorch by following the instructions
   [here](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#build-the-project)
2. A PyTorch install that supports [MPI as distributed communication
   backend](https://pytorch.org/docs/stable/distributed.html#backends-that-come-with-pytorch). You need to build and
   install PyTorch
   from source to include MPI as a backend.

Once above steps are complete, you can run MPT using the following commands:

```
# For running on a single gpu and benchmarking
PYTHONPATH=/mnt/work/FasterTransformer python scripts/inference/run_mpt_with_ft.py --ckpt_path mpt-ft-7b/1-gpu \
    --lib_path /mnt/work/FasterTransformer/build/lib/libth_transformer.so --time

# Run with -h to see various generation arguments
PYTHONPATH=/mnt/work/FasterTransformer python scripts/inference/run_mpt_with_ft.py -h

# Run on 2 gpus. You need to create an FT checkpoint for 2-gpus first.
# allow-run-as-root is only needed if you are running as root
PYTHONPATH=/mnt/work/FasterTransformer mpirun -n 2 --allow-run-as-root \
    python scripts/inference/run_mpt_with_ft.py \
    --ckpt_path mpt-ft-7b/2-gpu --lib_path /mnt/work/FasterTransformer/build/lib/libth_transformer.so --time

# Add prompts in a text file and generate text
echo "Write 3 reasons why you should train an AI model on domain specific data set." > prompts.txt
PYTHONPATH=/mnt/work/FasterTransformer python scripts/inference/run_mpt_with_ft.py \
    --ckpt_path mpt-ft-7b/1-gpu --lib_path /mnt/work/FasterTransformer/build/lib/libth_transformer.so \
    --sample_input_file prompts.txt --sample_output_file output.txt
```

## Running MPT with TensorRT-LLM

MPT-like architectures can be used with NVIDIA's [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) library for language model inference. To do so, follow the instructions in the [examples/mpt](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.6.1/examples/mpt) directory for the most recent release, which will show you how to:

1. Convert an MPT HuggingFace checkpoint into the FasterTransformer format.
2. Build a TensorRT engine with the FasterTransformer weights

Using this engine, you can utilize TensorRT-LLM for fast inference. If you would like to use TensorRT-LLM as an end-to-end solution for an inference service, you can utilize the built engine with an NVIDIA Triton server backend: an example server can be found in [this repository](https://github.com/triton-inference-server/tensorrtllm_backend/tree/v0.6.1) accompanying the most recent release.
