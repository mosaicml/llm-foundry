## In-context learning (ICL) evaluaton
What is this module for?


## ICL Tasks

ICL evaluation measures a model’s ability to solve novel problems by being provided examples in-context without ever being specifically trained to answer such questions.

Composer supports a number of different standard ICL formats and allows and allows users to upload their own datasets that correspond to those formats.

This document will explain the available composer ICL formats, give some basic pointers on adding new datasets in those formats, and catalog the datasets currently used by the research team to assess our hero runs.

## Supported ICL formats

Composer currently supports four ICL formats

1. [InContextLearningQATaskDataset](https://github.com/mosaicml/composer/blob/dev/composer/datasets/in_context_learning_evaluation.py#L92-L253)
2. [InContextLearningLMTaskDataset](https://github.com/mosaicml/composer/blob/dev/composer/datasets/in_context_learning_evaluation.py#L256-L402)
3. [InContextLearningMultipleChoiceTaskDataset](https://github.com/mosaicml/composer/blob/dev/composer/datasets/in_context_learning_evaluation.py#L405-L599)
4. [InContextLearningSchemaTaskDataset](https://github.com/mosaicml/composer/blob/dev/composer/datasets/in_context_learning_evaluation.py#L602-L773)

### InContextLearningQATaskDataset

The ICL question answering (QA) task supports free response question answering evaluation using the model’s generate function. A QA dataset consists of a list of JSONs containing a question (under the key “context”), a correct answer (under the key “answer”), and a list of alternative spellings of the answer that would be considered permissible (under the key “aliases”). The QA task works with the NLP metric: [InContextLearningQAAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningQAAccuracy.html)  which assigns a model's output to be "correct" if, conditioned on the context, the model's generate method produces a string that is a case-insensitive prefix for either the "answer" or any of the "aliases".

--------
Required keys for each datum:
* context: str
* answer: str
* aliases: List[str]
------

An example datum is below:

```jsx
{"context": "What star sign is Jamie Lee Curtis?", "answer": "Scorpio", "aliases": ["Scorpio", "Skorpio"]}
```

The QA task expects a **prompt** string, a **continuation delimiter** to separate questions from answers, an **example delimiter** to separate few shot examples from one another, and a **question prelimiter** to put before each question. If using the following settings, with 2 few shot examples in context, the above datum may be rendered to the model as:

```jsx
prompt_string: "Answer the following trivia question:\n", example_delimiter: "\n", continuation_delimiter: " Answer: ", question_prelimiter: "Question: "
```

> Answer the following trivia question:
Question: What is the Japanese share index called? Answer: Nikkei
Question: Who was the man behind The Chipmunks? Answer: David Seville
Question: What star sign is Jamie Lee Curtis? Answer:
> 

The model would then be expected to generate a series of tokens beginning with either of the aliases: Scorpio/Skorpio.

Below is a complete YAML that works with the TriviaQA datset in `local_data/triviaqa.jsonl`

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
    example_delimiter: '\n' # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>

### InContextLearningLMTaskDataset

The ICL language modeling (LM) task assesses the model’s ability to predict a precise sequence of tokens (called a continuation) following some context using the model’s `forward` function. An LM dataset consists of a list of JSONs containing a context (under the key “context”) and a continuation (under the key “continuation” that the model must correctly predict conditioned on the context. The LM task works with the NLP metric: [InContextLearningLMAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningLMAccuracy.html) which assigns a model's output to be "correct" if, conditioned on the context tokens, the model's argmax output logits exactly match the tokens in the continuation.

--------
Required keys for each datum:
* context: str
* continuation: str
------

An example datum is below:

```jsx
{"context": "With Tristran's next step he was standing beside a lake, and the candlelight shone brightly on the water; and then he was walking through the mountains, through lonely crags, where the candlelight was reflected in the eyes of the creatures of the high snows; and then he was walking through the clouds, which, while not entirely substantial, still supported his weight in comfort; and then, holding tightly to his candle, he was underground, and the candlelight glinted back at him from the wet cave walls; now he was in the mountains once more; and then he was on a road through wild forest, and he glimpsed a chariot being pulled by two goats, being driven by a woman in a red dress who looked, for the glimpse he got of her, the way Boadicea was drawn in his history books; and another step and he was in a leafy glen, and he could hear the chuckle of water as it splashed and sang its way into a small brook.\n\nHe took another step, but he was still in the", "continuation": "glen"}
```

The LM task expects a **prompt** string, a **continuation delimiter** to separate continuation from context and an **example delimiter** to separate few shot examples from one another. If using the following settings, with 0 few shot examples in context, the above datum may be rendered to the model as:

> With Tristran's next step he was standing beside a lake, and the candlelight shone brightly on the water; and then he was walking through the mountains, through lonely crags, where the candlelight was reflected in the eyes of the creatures of the high snows; and then he was walking through the clouds, which, while not entirely substantial, still supported his weight in comfort; and then, holding tightly to his candle, he was underground, and the candlelight glinted back at him from the wet cave walls; now he was in the mountains once more; and then he was on a road through wild forest, and he glimpsed a chariot being pulled by two goats, being driven by a woman in a red dress who looked, for the glimpse he got of her, the way Boadicea was drawn in his history books; and another step and he was in a leafy glen, and he could hear the chuckle of water as it splashed and sang its way into a small brook.
He took another step, but he was still in the
> 

The model would then be expected output “ glen”

Below is a complete YAML that works with the Lambada OpenAI datset in `local_data/lambada_openai.jsonl`

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
    example_delimiter: '\n' # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates contexts from continuations
>

### InContextLearningMultipleChoiceTaskDataset

The ICL multiplce choice (MC) task assesses the model’s ability to answer multiple choice questions by assigning highest per token probability to the correct answer. An MC dataset consists of a list of JSONs containing a query (under the key “query”), a list of choices (under the key “choices”), and the index indicating the correct answer (under the key "gold\_idx"). The MC task works with the NLP metric: [InContextLearningMultipleChoiceAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningMultipleChoiceAccuracy.html) which separately runs model `forward` on the query prepended to each choice, and then assigns the model to be correct if the correct choice has the lowest per token perplexity conditioned on the query.

--------
Required keys for each datum:
* query: str
* choices: str
* gold: int
------

An example datum is below:
```jsx
{"query": "High jump: A boy is running down a track. The boy", "choices": ["runs into a car.", "gets in a mat.", "lifts his body above the height of a pole.", "stands on his hands and springs."], "gold": 2}
```
The MC task expects a prompt string, a continuation delimiter to separate continuation from context and an example delimiter to separate few shot examples from one another. If using the following settings, with 0 few shot examples in context, the above datum may be rendered as four different inputs to the model:

> High jump: A boy is running down a track. The boy runs into a car.
> 
> High jump: A boy is running down a track. The boy gets in a mat.
> 
> High jump: A boy is running down a track. The boy lifts his body above the height of a pole.
>
> High jump: A boy is running down a track. The boy stands on his hands and springs.
> 

The model would be assigned correct if it assigns the lowest per token perplexity to the sequence " lifts his body above the height of a pole." 

Below is a complete YAML that works with the HellaSwag dataset in `local_data/hellaswag.jsonl`

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
    example_delimiter: '\n' # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>


### InContextLearningSchemaTaskDataset

The ICL schema task assesses the model’s ability to determine which of some set of possible contexts (under the key "context\_options") makes a sequence of tokens (under the key "continuation") most likely, with the correct context indicated by "gold\_idx". Based on: [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847)

The scgema task works with the NLP metric: [InContextLearningMultipleChoiceAccuracy](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.metrics.InContextLearningMultipleChoiceAccuracy.html) which separately runs model `forward` on each context option prepended to the continuation and rates the model correct if it assigns minimum per token perplexity to the continuation conditioned on the true context.

--------
Required keys for each datum:
* query: str
* choices: str
* gold: int
------

An example datum is below:
```jsx
{"context_options": ["Jim comforted Kevin because Jim", "Jim comforted Kevin because Kevin"], "continuation": "was so upset.", "gold": 1}
```
The Schema task expects a prompt string, a continuation delimiter to separate continuation from context and an example delimiter to separate few shot examples from one another. If using the following settings, with 0 few shot examples in context, the above datum may be rendered as two different inputs to the model:

> Jim comforted Kevin because Jim was so upset.
> 
> Jim comforted Kevin because Kevin was so upset.
> 


The model would be assigned correct if per token perplexity of the sequence " was so upset" is lower in the second version than it is in the first version.

Below is a complete YAML that works with the Winograd dataset in `local_data/winograd_wsc.jsonl`

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
    example_delimiter: '\n' # this goes between fewshot examples
    continuation_delimiter: ' ' # this separates questions from answers
>



