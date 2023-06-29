# In-context learning (ICL) evaluaton
This folder contains the MosaicML LLM evaluation suite. It is a [blazingly fast](https://www.mosaicml.com/blog/llm-evaluation-for-icl), multi-GPU-enabled ICL evaluaton suite with native [FSDP](https://pytorch.org/docs/stable/fsdp.html) compatibility with any model on the HuggingFace hub and any PyTorch model that implements the [`ComposerModel` interface](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.ComposerModel.html#composermodel). We also include collection of ICL datasets we refer to as our [Model Gauntlet](https://github.com/mosaicml/llm-foundry/blob/scripts/eval/local_data/MODEL_GAUNTLET.md) organized into 6 broad categories of competency that we expect good foundation models to have.

You can evaluate a model by preparing an evaluaton YAML following the format of the examples in the [`scripts/eval/yamls` directory](https://github.com/mosaicml/llm-foundry/tree/main/scripts/eval/yamls).

**Offline evaluation**
You can run the evaluation script on a model checkpoint via `composer eval/eval.py YOUR_YAML` from the `scripts` directory or launch it on the MosaicML platform using a an MCLI YAML following the format of [`llm-foundry/mcli/mcli-1b-eval.yaml`](https://github.com/mosaicml/llm-foundry/blob/main/mcli/mcli-1b-eval.yaml).
Your YAML must have a config section entitled `icl_tasks`, this can either be a list of dictionaries of the form

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


**Evaluation during training**
You can also add ICL evaluation to your training runs by adding an `icl_tasks` config to your training config at the same depth as the `model` subconfig.


----

## ICL Tasks

ICL evaluation measures a model’s ability to solve novel problems by being provided examples in-context without ever being specifically trained to answer such questions.

Composer supports a number of different standard ICL formats and allows users to upload their own datasets that correspond to those formats.

This document explains the ICL formats compatible with [Composer](https://github.com/mosaicml/composer), summarizes how to add new datasets in those formats, and catalogs the datasets currently used by the research team to evaluate models.

----

## Supported ICL formats

Composer currently supports four ICL formats

1. [InContextLearningQATaskDataset](https://github.com/mosaicml/composer/blob/v0.14.0/composer/datasets/in_context_learning_evaluation.py#L92-L253)
2. [InContextLearningLMTaskDataset](https://github.com/mosaicml/composer/blob/v0.14.0/composer/datasets/in_context_learning_evaluation.py#L256-L402)
3. [InContextLearningMultipleChoiceTaskDataset](https://github.com/mosaicml/composer/blob/v0.14.0/composer/datasets/in_context_learning_evaluation.py#L405-L599)
4. [InContextLearningSchemaTaskDataset](https://github.com/mosaicml/composer/blob/v0.14.0/composer/datasets/in_context_learning_evaluation.py#L602-L773)

--------

### InContextLearningQATaskDataset

The ICL question answering (QA) task supports free response question answering evaluation using the model’s generate function. A QA dataset consists of a list of JSONs containing a question (under the key `context`), a correct answer (under the key `answer`), and a list of alternative spellings of the answer that would be considered permissible (under the key `aliases`). The QA task works with the NLP metric: [InContextLearningQAAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningQAAccuracy.html) which assigns a model's output to be "correct" if, conditioned on the context, the model's generate method produces a string that is a normalized prefix for either the `answer` or any of the `aliases`.

Required keys for each datum:
* `context`: str
* `answer`: str
* `aliases`: List[str]

An example datum is below:

```jsx
{"context": "What star sign is Jamie Lee Curtis?", "answer": "Scorpio", "aliases": ["Scorpio", "Skorpio"]}
```

The QA task expects a **prompt string**, a **continuation delimiter** to separate questions from answers, an **example delimiter** to separate few shot examples from one another, and a **question prelimiter** to put before each question. If using the following settings, with 2 examples in context, the above datum may be rendered to the model as:

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
    icl_task_type: question_answering
    metric_names:
    - InContextLearningQAAccuracy
    prompt_string: '' # this goes at the beginning of each input
    example_delimiter: "\n" # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>

----

### InContextLearningLMTaskDataset

The ICL language modeling (LM) task assesses the model’s ability to predict a precise sequence of tokens (called a continuation) following some context using the model’s `forward` function. An LM dataset consists of a list of JSONs containing a context (under the key `context`) and a continuation (under the key `continuation` that the model must correctly predict conditioned on the context. The LM task uses the NLP metric [InContextLearningLMAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningLMAccuracy.html), which assigns a model's output to be "correct" if, conditioned on the context tokens, the model's argmax output logits exactly match the tokens in the continuation.

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

The ICL multiple choice (MC) task assesses the model’s ability to answer multiple choice questions by assigning highest per token probability to the correct answer. An MC dataset consists of a list of JSONs containing a query (under the key `query`), a list of choices (under the key `choices`), and the index indicating the correct answer (under the key `gold`). The MC task works with the NLP metric [InContextLearningMultipleChoiceAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningMultipleChoiceAccuracy.html), which separately runs the model's `forward()` method on the query prepended to each choice, and then determines the model to be correct if the correct choice has the lowest per token perplexity conditioned on the query.

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
    - InContextLearningMCExpectedCalibrationError
    prompt_string: '' # this goes at the beginning of each input
    example_delimiter: "\n" # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>

----

### InContextLearningSchemaTaskDataset

The ICL schema task assesses the model’s ability to determine which of some set of possible contexts (under the key `context_options`) makes a sequence of tokens (under the key `continuation`) most likely, with the correct context indicated by "gold". This task is based on [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847).

The schema task works with the NLP metric [InContextLearningMultipleChoiceAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningMultipleChoiceAccuracy.html), which separately runs the model's `forward()` method on each context option prepended to the continuation and rates the model correct if it assigns minimum per token perplexity to the continuation conditioned on the true context.

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
