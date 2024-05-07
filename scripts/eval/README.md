# In-context learning (ICL) evaluation

This folder contains the MosaicML LLM evaluation suite. It is a [blazingly fast](https://www.mosaicml.com/blog/llm-evaluation-for-icl), multi-GPU-enabled ICL evaluation suite with native [FSDP](https://pytorch.org/docs/stable/fsdp.html) compatibility with any model on the HuggingFace hub and any PyTorch model that implements the [`ComposerModel` interface](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.ComposerModel.html#composermodel). We also include collection of ICL datasets we refer to as our [Eval Gauntlet](./local_data/EVAL_GAUNTLET.md) organized into 6 broad categories of competency that we expect good foundation models to have.

You can evaluate a model by preparing an evaluation YAML following the format of the examples in the [`scripts/eval/yamls` directory](https://github.com/mosaicml/llm-foundry/tree/main/scripts/eval/yamls).

----

## Quickstart

### Offline evaluation

To run offline evaluation, download this repo and run the following commands:

<!--pytest.mark.skip-->
```bash
cd llm-foundry/scripts
composer eval/eval.py eval/yamls/hf_eval.yaml
```

This will run `EleutherAI/gpt-neo-125m` through the MosaicML Eval Gauntlet, a diverse evaluation suite consisting of over 30 benchmarks. You can update the configuration directly in the `hf_eval.yaml` YAML file, or override the values in the YAML with CLI args, such as:

<!--pytest.mark.skip-->
```bash
cd llm-foundry/scripts
composer eval/eval.py eval/yamls/hf_eval.yaml \
    model_name_or_path=mosaicml/mpt-7b
```

You can also modify the specific benchmarks executed and their formatting by modifying the contents of `tasks.yaml` and you can modify the choice of composite scores and the set of tasks they consist of by modifying `eval_gauntlet_v0.2.yaml`.


### Evaluation during training
To run evaluation during training, download this repo, follow the instructions in `scripts/train/README.md` to perform single node pre-training and run the following commands

<!--pytest.mark.skip-->
```bash
cd llm-foundry/scripts/train
composer train.py yamls/pretrain/mpt-125m_eval.yaml train_loader.dataset.split=train_small eval_loader.dataset.split=val_small
```
You can also modify the specific benchmarks executed and their formatting by modifying the contents of `tasks.yaml` and you can modify the choice of composite scores and the set of tasks they consist of by modifying `eval_gauntlet_v0.2.yaml`. You can also choose to either run the full evaluation or run on a subset number of batches per benchmark by setting `icl_subset_num_batches`.

----
## In-depth walkthrough

ICL evaluation can be done offline via the `scripts/eval/eval.py` or during training via `scripts/train/train.py`.

In order to do ICL evaluation you must specify a set of benchmarks you'd like to run via the `icl_tasks` key in your eval/training config. `icl_tasks` can either consist of config, or it can be a file path pointing to a locally accessible YAML config (see `scripts/eval/yamls/tasks.yaml` for an example).


#### ICL task YAML format
Your YAML must have a config section entitled `icl_tasks` specifying the benchmarks to evaluate againts, this can either be a list of dictionaries of the form

```jsx
icl_tasks:
  -
    label: piqa
    dataset_uri: # ADD YOUR OWN DATASET URI
    num_fewshot: [5]
    icl_task_type: multiple_choice
    continuation_delimiter: ' '
    example_delimiter: "\n"
    prompt_string: ''
  -
    label: lambada
    dataset_uri: # ADD YOUR OWN DATASET URI
    num_fewshot: [0]
    icl_task_type: language_modeling
```

or a local path pointing to a YAML containing an icl\_tasks config.

Note that if continuation\_delimiter, example\_delimiter, or prompt\_string are omitted they will default to the values below:

```jsx
continuation_delimiter: ' '
example_delimiter: "\n"
prompt_string: ''
```

#### Eval gauntlet YAML format
Your YAML may optionally have a config section entitled eval\_gauntlet specifying how to aggregate the results (if absent, only the individual benchmark accuracies will be reported). After the tasks listed in the `icl_tasks` config are evaluated, the eval script will use the `eval_gauntlet` config, if specified, to aggregate the individual benchmarks into composite scores.


An `eval_gauntlet` config  must specify the list of categories you'd like to generate composite scores for, as well as the list of benchmarks to be included in each category. For each category you need to list the name and the `num_fewshot`. These two values must exactly match the values specified in the `icl_tasks` config. Additionally you must specify the random baseline accuracy for each benchmark.

There are also three flags indicating how to perform the aggregation:
1. `weighting` can either be `EQUAL` (all tasks are weighted equally), `SAMPLE_SZ` (tasks are weighted proportional to the size of the dataset), or `LOG_SAMPLE_SZ` (tasks are weighted proportional to the logarithm of the dataset size).
2. `subtract_random_baseline` can either be `true` or `false`. If `true` we will subtract the random baseline accuracy from the final accuracy before averaging, otherwise it will be averaged in as is.
3. `rescale_accuracy` can either be `true` or `false`. If `true` (and if `subtract_random_baseline` was also `true`), the accuracy will be rescaled to be <= 1 before averaging.

An example config is below:
```jsx
eval_gauntlet:
  weighting: EQUAL
  subtract_random_baseline: true
  rescale_accuracy: true
  categories:
  - name: world_knowledge
    benchmarks:
    - name: jeopardy
      num_fewshot: 10
      random_baseline: 0
    - name: mmlu
      num_fewshot: 10
      random_baseline: 0.25
  - name: language_understanding
    benchmarks:
    - name: lambada_openai
      num_fewshot: 0
      random_baseline: 0.0
    - name: hellaswag
      num_fewshot: 10
      random_baseline: 0.25
```

You can either specify your `eval_gauntlet` config directly in your eval/train YAML or via a local path pointing to a YAML containing an `eval_gauntlet` config.


### Offline evaluation

You can run the evaluation script on a model checkpoint via `composer eval/eval.py YOUR_YAML` from the `scripts` directory or launch it on the MosaicML platform using a an MCLI YAML following the format of [`llm-foundry/mcli/mcli-1b-eval.yaml`](https://github.com/mosaicml/llm-foundry/blob/main/mcli/mcli-1b-eval.yaml).

You can use the default `icl_tasks` and `eval_gauntlet` configs or specify your own following the instructions above.

### Evaluation during training

You can use ICL evaluation during training by taking an ordinary training YAML and adding an `icl_tasks` and `eval_gauntlet` config. You should also specify `icl_seq_len` in your training YAML and you can optionally run a truncated version of eval on a random subset of each benchmark by specifying a value for `icl_subset_num_batches`

An example is given below:
```
  icl_tasks: eval/yamls/tasks.yaml # or use tasks_light.yaml
  icl_subset_num_batches: 100 # -1, or omit this key entirely, to evaluate on all batches
  eval_gauntlet: 'eval/yamls/eval_gauntlet_v0.2.yaml'
  icl_seq_len: 1024
```

For training, we recommend you do not run the full eval gauntlet. Instead either use the `tasks_light.yaml` which is a subset of the full gauntlet benchmarks, or set `icl_subset_num_batches` to a small number O(100) which will only run each benchmark on a random sample of `icl_subset_num_batches` batches.

You can use the default `icl_tasks` and `eval_gauntlet` configs or specify your own following the instructions above.

----

## ICL Tasks

ICL evaluation measures a model’s ability to solve novel problems by being provided examples in-context without ever being specifically trained to answer such questions.

We support a number of standard ICL formats and allow users to upload their own datasets that correspond to these formats. All of our ICL task types are implemented in `llm-foundry/llmfoundry/eval/datasets/in_context_learning_evaluation.py` while all of our ICL
metrics are implemented in `llm-foundry/llmfoundry/eval/metrics/nlp.py`. You can see which metrics work with which task types in the `llmfoundry.utils.builders.build_icl_evaluators` helper function.

This document explains the ICL formats compatible with [Composer](https://github.com/mosaicml/composer), summarizes how to add new datasets in those formats, and catalogs the datasets currently used by the research team to evaluate models.

----

## Supported ICL formats

llm-foundry currently supports five ICL formats:

1. InContextLearningGenerationTaskWithAnswersDataset
2. InContextLearningLMTaskDataset
3. InContextLearningMultipleChoiceTaskDataset
4. InContextLearningSchemaTaskDataset
5. InContextLearningCodeEvalDataset

----

### InContextLearningGenerationTaskWithAnswersDataset

The ICL generation with answers task supports free response generation evaluation using the model’s generate function. A generation dataset consists of a list of JSONs containing a prompt (under the key `context`), a correct answer (under the key `answer`), and a list of alternative answers that would be considered permissible (under the key `aliases`). The generation task works with the NLP metric: InContextLearningGenerationExactMatchAccuracy which assigns a model's output to be "correct" if, conditioned on the context, the model's generate method produces a string that is a normalized prefix for either the `answer` or any of the `aliases`.

Required keys for each datum:
* `context`: str
* `answer`: str
* `aliases`: List[str]

An example datum is below:

```jsx
{"context": "What star sign is Jamie Lee Curtis?", "answer": "Scorpio", "aliases": ["Scorpio", "Skorpio"]}
```

The generation task expects a **prompt string**, a **continuation delimiter** to separate questions from answers, an **example delimiter** to separate few shot examples from one another, and a **question prelimiter** to put before each question. If using the following settings, with 2 examples in context, the above datum may be rendered to the model as:

```jsx
prompt_string: "Answer the following trivia question:\n", example_delimiter: "\n", continuation_delimiter: " Answer: ", question_prelimiter: "Question: "
```

> Answer the following trivia question:
Question: What is the Japanese share index called? Answer: Nikkei
Question: Who was the man behind The Chipmunks? Answer: David Seville
Question: What star sign is Jamie Lee Curtis? Answer:
>

The model would then be expected to generate a series of tokens beginning with either of the aliases: `Scorpio/Skorpio`.

Below is a complete YAML section that works with the TriviaQA dataset in [`scripts/eval/local_data/triviaqa.jsonl`](https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/local_data/triviaqa.jsonl):

>
    label: triviaqa
    dataset_uri: local_data/triviaqa.jsonl
    num_fewshot:
    - 0
    - 1
    - 5
    - 10
    batch_size: 4
    icl_task_type: generation_task_with_answers
    metric_names:
    - InContextLearningGenerationExactMatchAccuracy
    prompt_string: '' # this goes at the beginning of each input
    example_delimiter: "\n" # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>

----

### InContextLearningLMTaskDataset

The ICL language modeling (LM) task assesses the model’s ability to predict a precise sequence of tokens (called a continuation) following some context using the model’s `forward` function. An LM dataset consists of a list of JSONs containing a context (under the key `context`) and a continuation (under the key `continuation` that the model must correctly predict conditioned on the context. The LM task uses the NLP metric InContextLearningLMAccuracy, which assigns a model's output to be "correct" if, conditioned on the context tokens, the model's argmax output logits exactly match the tokens in the continuation.

Required keys for each datum:
* `context`: str
* `continuation`: str


An example datum is below:

```jsx
{"context": "With Tristran's next step he was standing beside a lake, and the candlelight shone brightly on the water; and then he was walking through the mountains, through lonely crags, where the candlelight was reflected in the eyes of the creatures of the high snows; and then he was walking through the clouds, which, while not entirely substantial, still supported his weight in comfort; and then, holding tightly to his candle, he was underground, and the candlelight glinted back at him from the wet cave walls; now he was in the mountains once more; and then he was on a road through wild forest, and he glimpsed a chariot being pulled by two goats, being driven by a woman in a red dress who looked, for the glimpse he got of her, the way Boadicea was drawn in his history books; and another step and he was in a leafy glen, and he could hear the chuckle of water as it splashed and sang its way into a small brook.\n\nHe took another step, but he was still in the", "continuation": " glen"}
```

The LM task expects a **prompt string**, a **continuation delimiter** to separate continuation from context, and an **example delimiter** to separate few shot examples from one another. If using the following settings, with 0 examples in context, the above datum may be rendered to the model as:

> With Tristran's next step he was standing beside a lake, and the candlelight shone brightly on the water; and then he was walking through the mountains, through lonely crags, where the candlelight was reflected in the eyes of the creatures of the high snows; and then he was walking through the clouds, which, while not entirely substantial, still supported his weight in comfort; and then, holding tightly to his candle, he was underground, and the candlelight glinted back at him from the wet cave walls; now he was in the mountains once more; and then he was on a road through wild forest, and he glimpsed a chariot being pulled by two goats, being driven by a woman in a red dress who looked, for the glimpse he got of her, the way Boadicea was drawn in his history books; and another step and he was in a leafy glen, and he could hear the chuckle of water as it splashed and sang its way into a small brook.
He took another step, but he was still in the
>

The model would then be expected output “ glen”.

Below is a YAML section that works with the Lambada OpenAI dataset in [`scripts/eval/local_data/lambada_openai.jsonl`](https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/local_data/lambada_openai.jsonl):

>
    label: lambada_openai
    dataset_uri: local_data/lambada_openai.jsonl
    num_fewshot:
    - 0
    batch_size: 4
    icl_task_type: language_modeling
    metric_names:
    - InContextLearningLMAccuracy
    prompt_string: '' # this goes at the beginning of each input
    example_delimiter: "\n" # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates contexts from continuations
>

----

### InContextLearningMultipleChoiceTaskDataset

The ICL multiple choice (MC) task assesses the model’s ability to answer multiple choice questions by assigning highest per token probability to the correct answer. An MC dataset consists of a list of JSONs containing a query (under the key `query`), a list of choices (under the key `choices`), and the index indicating the correct answer (under the key `gold`). The MC task works with the NLP metric InContextLearningMultipleChoiceAccuracy, which separately runs the model's `forward()` method on the query prepended to each choice, and then determines the model to be correct if the correct choice has the lowest per token perplexity conditioned on the query.

Required keys for each datum:
* `query`: str
* `choices`: str
* `gold`: int

An example datum is below:
```jsx
{"query": "High jump: A boy is running down a track. The boy", "choices": ["runs into a car.", "gets in a mat.", "lifts his body above the height of a pole.", "stands on his hands and springs."], "gold": 2}
```
The MC task expects a **prompt string**, a **continuation delimiter** to separate continuation from context, and an **example delimiter** to separate few shot examples from one another. If using the following settings, with 0 examples in context, the above datum may be rendered as four different inputs to the model:

> High jump: A boy is running down a track. The boy runs into a car.
>
> High jump: A boy is running down a track. The boy gets in a mat.
>
> High jump: A boy is running down a track. The boy lifts his body above the height of a pole.
>
> High jump: A boy is running down a track. The boy stands on his hands and springs.
>

The model would be deemed correct if it assigns the lowest per token perplexity to the sequence " lifts his body above the height of a pole."

Below is a YAML section that works with the HellaSwag dataset in [`scripts/eval/local_data/hellaswag.jsonl`](https://raw.githubusercontent.com/mosaicml/llm-foundry/main/scripts/eval/local_data/hellaswag.jsonl):

>
    label: hellaswag
    dataset_uri: local_data/hellaswag.jsonl # ADD YOUR OWN DATASET URI
    num_fewshot:
    - 0
    - 1
    - 5
    - 10
    batch_size: 4
    icl_task_type: multiple_choice
    metric_names:
    - InContextLearningMultipleChoiceAccuracy
    prompt_string: '' # this goes at the beginning of each input
    example_delimiter: "\n" # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>

----

### InContextLearningSchemaTaskDataset

The ICL schema task assesses the model’s ability to determine which of some set of possible contexts (under the key `context_options`) makes a sequence of tokens (under the key `continuation`) most likely, with the correct context indicated by "gold". This task is based on [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847).

The schema task works with the NLP metric InContextLearningMultipleChoiceAccuracy, which separately runs the model's `forward()` method on each context option prepended to the continuation and rates the model correct if it assigns minimum per token perplexity to the continuation conditioned on the true context.

Required keys for each datum:
* query: str
* choices: str
* gold: int

An example datum is below:
```jsx
{"context_options": ["Jim comforted Kevin because Jim", "Jim comforted Kevin because Kevin"], "continuation": "was so upset.", "gold": 1}
```
The Schema task expects a **prompt string**, a **continuation delimiter** to separate continuation from context, and an **example delimiter** to separate few shot examples from one another. If using the following settings, with 0 few shot examples in context, the above datum may be rendered as two different inputs to the model:

> Jim comforted Kevin because Jim was so upset.
>
> Jim comforted Kevin because Kevin was so upset.
>


The model would be assigned correct if per token perplexity of the sequence " was so upset" is lower in the second version than it is in the first version.

Below is a YAML section that works with the Winograd dataset in [`scripts/eval/local_data/winograd_wsc.jsonl`](https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/local_data/winograd_wsc.jsonl):

>
    label: winograd
    dataset_uri: local_data/winograd_wsc.jsonl
    num_fewshot:
    - 0
    - 1
    - 5
    - 10
    batch_size: 4
    icl_task_type: schema
    metric_names:
    - InContextLearningMultipleChoiceAccuracy
    - InContextLearningMCExpectedCalibrationError
    prompt_string: '' # this goes at the beginning of each input
    example_delimiter: "\n" # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>


----

### InContextLearningCodeEvalDataset

The ICL CodeEvalDataset takes a prompt, and, working with the NLP metric [InContextLearningCodeEvalAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningCodeEvalAccuracy.html), generates code which gets run against the supplied tests, as in HumanEval ([Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)) and MBPP ([Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)). This generation involves many decoding steps, so can take longer per sample than other ICL tasks. An example datum:

```json
{"task_id": "JavaScript/2", "prompt": "/* Given a positive floating point number, it can be decomposed into\n  and integer part (largest integer smaller than given number) and decimals\n  (leftover part always smaller than 1).\n\n  Return the decimal part of the number.\n  >>> truncateNumber(3.5)\n  0.5\n  */\nconst truncateNumber = (number) => {\n", "canonical_solution": "  return number % 1.0;\n}\n\n", "test": "const testTruncateNumber = () => {\n  console.assert(truncateNumber(3.5) === 0.5)\n\n  console.assert(Math.abs(truncateNumber(1.33) - 0.33) < 1e-6)\n\n  console.assert(Math.abs(truncateNumber(123.456 - 0.456) < 1e-6))\n}\n\ntestTruncateNumber()\n", "entry_point": "truncateNumber", "test_inputs": ["3.5", "1.33", "123.456"], "test_outputs": ["0.5", "0.33", "0.456"], "language": "javascript"}
```

Required keys for each datum:

* `prompt: str`
* `test: str`
* `entry_point: str`
* `test_inputs: List[str]`
* `test_outputs: List[str]`
* `language: str`

Code evaluation can happen locally (insecure) or inside an AWS Lambda function sandbox. This is controlled by setting the environment variable `CODE_EVAL_DEVICE` to `LOCAL` or `LAMBDA`. If set to `LAMBDA`, you must also provide `CODE_EVAL_URL` and `CODE_EVAL_APIKEY` to query the API gateway in the AWS Sandbox.

----

### Build your own dataset (BYOD)
Building a dataset compatible with our eval suite is very easy if it fits with one of the four supported task types. Simply choose the appropriate task type (LM, MC, QA, or Schema) and process each dataset into a jsonl format in which each row has the format described above.

Below is a minimal script which prepares the [Winograd schema challenge](https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf) hosted on [HuggingFace](https://huggingface.co/datasets/winograd_wsc). This script can be modified to generate other datasets based on the HuggingFace dataset hub.
```python
from datasets import load_dataset

upper_pronouns = [
        "A",
        "An",
        "The",
        "She",
        "He",
        "It",
        "They",
        "My",
        "His",
        "Her",
        "Their",
    ]

def __normalize_option(doc, option):
        # this function adapted from EleutherAI/lm-evaluation-harness

        # Append `'s` to possessive determiner based options.
        if doc["pronoun"].lower() in ["my", "his", "her", "our", "their"]:
            option += "'s"
        # Appropriately lowercase the pronoun in the option.
        pronoun = option.split()[0]
        start_of_sentence = doc["text"][doc["pronoun_loc"] - 2] == "."
        if not start_of_sentence and pronoun in upper_pronouns:
            return option.replace(pronoun, pronoun.lower())
        return option

def lower_first_letter(s):
    return s[0:1].lower() + s[1:]

def prep_winograd_wsc(row):
    # this function adapted from EleutherAI/lm-evaluation-harness

    prefix = row['text'][:row['pronoun_loc']]
    continuation = row['text'][row['pronoun_loc'] + len(row['pronoun']):]

    context_options = [
        prefix + __normalize_option(row, o) for o in row['options']
    ]

    return {
        "context_options": context_options,
        "continuation": continuation,
        "gold": row['label']
    }

def prep_dataset(out_file):
        dataset_name = ('winograd_wsc', 'wsc273')
        dataset = load_dataset(*dataset_name)

        with open(out_file, "w", encoding='utf8') as f:

            if dataset_name[0] == 'winogrande':
                split = dataset['validation']
            else:
                split = dataset['test'] if 'test' in dataset \
                    else dataset['validation']
            for row in split:
                row = prep_winograd_wsc(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Similarly, you can compile a dataset directly from [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) by modifying the script below:
```python
def prep_triviaqa(row):

    return {
        "context": f"Question: {row['question']}\nAnswer:",
        "answer": row['answer']['value'],
        "aliases": row['answer']['aliases']
    }

def prep_dataset(out_file):
    task = lm_eval_tasks.get_task_dict(['triviaqa'])['triviaqa']

    if task.has_test_docs():
        task_doc_func = task.test_docs
        task_set = "test"
    elif task.has_validation_docs():
        task_set = "val"
        task_doc_func = task.validation_docs
    with open(out_file, "w", encoding='utf8') as f:
            for row in task_doc_func():
                row = prep_triviaqa(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

#### A note on delimiters and tokenizers

When formatting samples, `prompt_string` is prepended to the beginning, then `num_fewshot` examples from the dataset are concatenated. Each few shot example is formatted with the context/continuation of each being separated by `continuation_delimiter`, then each example is separated from the others by the `example_delimiter`. Finally, we append the context/query/question/context options of the current sample to be evaluated and the `continuation_delimiter`.

Thus the structure of each question's preamble is `prompt | few shot examples | context | continuation delimiter`. The continuation (aka choices for MC) is then tokenized separately and the tokens of the preamble and tokens of the continuation are concatenated. It is important to note that if the continuation delimiter has a trailing space, it is stripped and instead prepended to the continuation. Furthermore, if the continuation does not have a leading space, one will be prepended.

----

# New Features

## Dynamic Padding
Add `trim_batch` to model's yaml similar to below:

```yaml
models:
-
  trim_batch: true
  ...
```

This will dynamically pad the batch to the maximum sequence length in the batch in contrast to padding to the maximum sequence length in the config. Obtains a speedup up to 10x for small batch sizes.

## Quantization Support
You can use the quantization features in the 🤗 Transformers. Both loading pre-quantized models and quantizing models.

### Loading pre-quantized models
Provide a 🤗 Hugging Face Hub model which includes `quantization_config` in the `config.json`.

```yaml
precision: "amp_bf16"

model_name_or_path: unsloth/llama-3-8b-bnb-4bit

models:
-
  model_name: ${model_name_or_path}
  model:
    name: hf_causal_lm
    pretrained_model_name_or_path: ${model_name_or_path}
    pretrained: true
    ...
...
```

### Quantizing models
Add the `quantization_config` to the model's yaml containing the parameters supported by the [🤗 Transformers quantization:](https://huggingface.co/docs/transformers/quantization)

```yaml
precision: "amp_bf16"

model_name_or_path: meta-llama/Meta-Llama-3-8B

models:
-
  model_name: ${model_name_or_path}
  model:
    name: hf_causal_lm
    pretrained_model_name_or_path: ${model_name_or_path}
    pretrained: true
    "quantization_config":
        "_load_in_4bit": true
        "_load_in_8bit": false
        "bnb_4bit_compute_dtype": "bfloat16"
        "bnb_4bit_quant_storage": "uint8"
        "bnb_4bit_quant_type": "nf4"
        "bnb_4bit_use_double_quant": true
        "llm_int8_enable_fp32_cpu_offload": false
        "llm_int8_has_fp16_weight": false
        "llm_int8_skip_modules": null
        "llm_int8_threshold": 6.0
        "load_in_4bit": true
        "load_in_8bit": false
        "quant_method": "bitsandbytes"
    ...
...
```

### Custom GPTQ Configuration
Although basic GPTQ is supported in the 🤗 Transformers, for more naunced control over its configuration in `llm-foundry/llmfoundry/utils/gptq.py` you can use the `gptq_config` instead:

```yaml
precision: "amp_bf16"

model_name_or_path: meta-llama/Meta-Llama-3-8B

models:
-
  model_name: ${model_name_or_path}
  model:
    name: hf_causal_lm
    pretrained_model_name_or_path: ${model_name_or_path}
    pretrained: true
    gptq_config:
      model_id: ${model_name_or_path}
      wbits: 4
      gs: 64
      actorder: True
      suffix: ${suffix}
      dataset: dataset.txt
      chunked: true
      chunk_size: 1024
      clip: True
      mse: False
    ...
...
```