# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from typing import Callable, Optional, Tuple

import openai
import requests
from tqdm import tqdm
from transformers import pipeline

# src https://github.com/lm-sys/FastChat/blob/8e38141ff5dd15f3138ccfd312dd73a471e986a1/fastchat/eval/table/prompt.jsonl
VICUNA_RUBRIC = "[Question]\n{{task}}\n\n[The Start of Assistant 1's Answer]\n{{model1}}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{{model2}}\n\n[The End of Assistant 2's Answer]\n\n[System]\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.\n\n"
VICUNA_SYSTEM = 'You are a helpful and precise assistant for checking the quality of the answer.'

# This is from the appendix of [Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
LIMA_score_prompt = """You are evaluating a response that has been submitted for a particular task, using a specific set of standards. Below is the data:
[BEGIN DATA]
***
[Task]: {task}
***
[Submission]: {submission}
***
[Criterion]: helpfulness:
"1": "Not helpful - The generated text is completely irrelevant, unclear, or incomplete. It does not provide any useful information to the user."
"2": "Somewhat helpful - The generated text has some relevance to the user’s question, but it may be unclear or incomplete. It provides only
partial information, or the information provided may not be useful for the user’s needs."
"3": "Moderately helpful - The generated text is relevant to the user’s question, and it provides a clear and complete answer. However, it may
lack detail or explanation that would be helpful for the user."
"4": "Helpful - The generated text is quite relevant to the user’s question, and it provides a clear, complete, and detailed answer. It offers
additional information or explanations that are useful for the user. However, some of the points of the response are somewhat repetitive or could
be combined for greater clarity and concision"
"5": "Very helpful - The generated text is highly relevant to the user’s question, and it provides a clear, complete, and detailed answer. It offers
additional information, explanations, or analogies that are not only useful but also insightful and valuable to the user. However, the structured
of the response is not well-organized and there is no clear progression or logical sequence of different points in the response."
"6": "Highly helpful - The generated text provides a clear, complete, and detailed answer. It offers additional information or explanations that
are not only useful but also insightful and valuable to the user. The response is also in a logical and easy-to-follow manner by explicitly using
headings, bullet points, or numbered lists to break up the information and make it easier to read."
***
[END DATA]
Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your
conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the choice only from “1, 2, 3, 4, 5, 6” (without quotes
or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the selected choice again by itself on a new line."""

LIMA_compare_prompt = """Imagine that you have a super-intelligent AI assistant, and that you require help with the following question. Which answer best satisfies
your needs?
Question: {{task}}
Answer A:
{{model1}}
Answer B:
{{model2}}
Comparing these two answers, which answer is better?
◼ Answer A is significantly better.
◼ Answer B is significantly better.
◼ Neither is significantly better."""


class CompletionFunction:
    """An abstraction of str->str.

    Works over either URLs, transformer pipelines, or files of input->output.

    For example,

    CompletionFunction.from_url(
        url='https://api.openai.com/v1/completions',
        model='text-davinci-003',
    )
    will call GPT3.5 on the prompt.
    """

    def __init__(self, completion_function, name: str = None):
        self.completion_function = completion_function
        self.name = name if name else completion_function.__name__

    def __call__(self, prompt: str):
        return self.completion_function(prompt)

    @classmethod
    def from_url(
        cls,
        url: str,
        model: str = None,
        system_message: str = None,
    ):
        if 'openai' in url and (name in model
                                for name in ['gpt-4', 'gpt-3.5-turbo']):
            print(
                'Using GPT-4 or GPT-3.5-turbo, must use chat completion endpoint'
            )
            cf = lambda prompt: openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_message
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    },
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0,  # vicuna does 0.2??
            ).choices[0].message
        elif 'openai' in url:
            cf = lambda prompt: openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0,
            ).choices[0].text
        else:
            print('Using non-openai API, support is limited')
            cf = lambda prompt: requests.post(url, json={
                'inputs': prompt
            }).json()['completion']
        return cls(cf, model if model else url)

    @classmethod
    def from_pipeline(cls, pipeline: pipeline, name: str = None):
        return cls(
            lambda prompt: pipeline(prompt, max_new_tokens=512)[0][
                'generated_text'], name)

    @classmethod
    def from_file(cls, file: str, name: str = None):
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(l) for l in lines]
        gpt_dict = {}
        for l in lines:
            gpt_dict[l['prompt']] = l['response']

        return cls(lambda prompt: gpt_dict[prompt], name=name)


class GPT4Judge(CompletionFunction):
    """A judge that uses GPT-4 to evaluate 1 or more models' output."""

    def build_prompt(self, task: str, model_outputs: Tuple, rubric: str = None):
        raise NotImplementedError

    def __call__(self, task: str, model_outputs: Tuple, rubric: str = None):
        return self.completion_function(self.build_prompt(task, model_outputs))

    @classmethod
    def from_defaults(
            cls,
            url: str = 'https://api.openai.com/v1/chat/completions',
            model: str = 'gpt-4',
            system_message: str = 'You judge fairly and explain your reasoning.'
    ):
        return cls.from_url(url, model, system_message)


class GPT4ComparisonJudge(GPT4Judge):

    def build_prompt(self,
                     task: str,
                     model_outputs: Tuple[str, str],
                     rubric: str = None):
        if rubric:
            # template in task and model_outputs to rubric
            rubric = rubric.replace('{{task}}', task)
            rubric = rubric.replace('{{model1}}', model_outputs[0])
            rubric = rubric.replace('{{model2}}', model_outputs[1])
            return rubric
        s = 'Help me to fairly judge the following responses to an instruction. Begin by explaining your reasoning. '
        s += "Then, finish by saying either 'The first response is better' or 'The second response is better'.\n\n"
        s += f'The instructions:\n{task}\n\n'
        s += f'The first response:\n{model_outputs[0]}\n\n'
        s += f'The second response:\n{model_outputs[1]}\n\n'
        return s

    @staticmethod
    def get_score(judge_outputs: str):
        if 'the first response is better' in judge_outputs.lower():
            return 0
        elif 'the second response is better' in judge_outputs.lower():
            return 1
        else:
            warnings.warn(f'judge_outputs = {judge_outputs}')


class VicunaJudge(GPT4ComparisonJudge):

    @classmethod
    def from_defaults(cls, *args, **kwargs):
        return cls.from_url(*args, system_message=VICUNA_SYSTEM, **kwargs)

    def build_prompt(self,
                     task: str,
                     model_outputs: Tuple[str, str],
                     _: str = None):
        return super().build_prompt(task, model_outputs, VICUNA_RUBRIC)

    @staticmethod
    def get_score(judge_outputs: str):
        # src
        # https://github.com/lm-sys/FastChat/blob/8e38141ff5dd15f3138ccfd312dd73a471e986a1/fastchat/eval/eval_gpt_review.py#L47
        try:
            score_pair = judge_outputs.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
            else:
                raise Exception('Invalid score pair.')
        except Exception as e:
            print(
                f'{e}\nContent: {judge_outputs}\nYou must manually fix the score pair.'
            )
            return [-1, -1]


class GPT4ScoreJudge(GPT4Judge):

    def build_prompt(self, task: str, model_outputs: Tuple[str]):
        return LIMA_score_prompt.format(task=task, submission=model_outputs[0])

    @staticmethod
    def get_score(judge_outputs: str):
        # return int(judge_outputs[-1])
        judge_outputs = judge_outputs.strip()
        # we need to clean up whitespace and punctuation then find the 1-6 score
        if judge_outputs[-1] in '123456':
            return int(judge_outputs[-1])
        # check for punctuation and whitespace
        elif judge_outputs[-2] in '123456':
            return int(judge_outputs[-2])
        else:
            warnings.warn(f'judge_outputs = {judge_outputs}')


LIMAScoreJudge = GPT4ScoreJudge


class LIMACompareJudge(GPT4ComparisonJudge):

    def build_prompt(self,
                     task: str,
                     model_outputs: Tuple[str, str],
                     _: str = None):
        return LIMA_compare_prompt.format(task=task,
                                          model_outputs=model_outputs)

    @staticmethod
    def get_score(judge_outputs: str):
        judge_outputs = judge_outputs.strip().lower()
        if 'answer a is significantly better' in judge_outputs:
            return 0
        elif 'answer b is significantly better' in judge_outputs:
            return 1
        else:
            return 0.5


class GEval:

    def __init__(
        self,
        judge: CompletionFunction,
        model1: CompletionFunction,
        rubric: Optional[str] = None,
        model2: Optional[CompletionFunction] = None,
        score_model: Optional[Callable] = None,
    ):
        self.judge = judge
        self.model1 = model1
        self.rubric = rubric
        self.model2 = model2
        self.score_model = score_model if score_model else judge.get_score

    def get_model_outputs(self, prompt: str):
        if not self.model2:
            return (self.model1(prompt),)
        else:
            return self.model1(prompt), self.model2(prompt)

    def get_judge_outputs(self, prompt: str):
        return self.judge(prompt, self.get_model_outputs(prompt), self.rubric)


class EvalScoreRunner:

    def __init__(self, g_eval: GEval, input_file: str, output_file: str):
        self.g_eval = g_eval
        self.input_file = input_file
        self.output_file = output_file

    def run(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(l) for l in lines]

        scores = []
        with open(self.output_file, 'w') as f:
            for l in tqdm(lines):
                prompt = l['prompt']
                model_outputs = self.g_eval.get_model_outputs(prompt)
                judge_outputs = self.g_eval.get_judge_outputs(prompt)
                score = self.g_eval.score_model(judge_outputs)
                l['model_outputs'] = model_outputs
                l['judge_outputs'] = judge_outputs
                l['score'] = score
                l['model'] = self.g_eval.model1.name
                l['judge'] = self.g_eval.judge.model  # model is a string for OpenAI models
                scores.append(score)
                f.write(json.dumps(l) + '\n')

        print(f'Average score: {sum(scores) / len(scores)}')


class EvalCompareRunner:
    """Run an evaluation of two models on a jsonl file of prompts.

    ```python
    # example of comparing a local model to GPT-3.5

    from transformers import pipeline
    from scripts.eval.geval.g_eval import EvalCompareRunner, GPT3_5CompareGEval


    example_pipe = pipeline('text-generation', model='local_model', tokenizer='local_model', device=0)
    runner = EvalCompareRunner(
        g_eval=GPT3_5CompareGEval(
            model1=CompletionFunction.from_pipeline(example_pipe)
        ),
        input_file='evals/my_eval_data.jsonl',
        output_file='runs/my_eval/compare_results.jsonl',
    )
    runner.run()
    ```

    Args:
        g_eval: The GEval object to use for evaluation
        input_file: The jsonl file of prompts
        output_file: The jsonl file to write the results to

    Returns:
        None
    """

    def __init__(self, g_eval: GEval, input_file: str, output_file: str):
        self.g_eval = g_eval
        self.input_file = input_file
        self.output_file = output_file

    def win_rate(self, model_num: int = 0):
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(l) for l in lines]
        wins = 0
        for l in lines:
            if l['winner'] == model_num:
                wins += 1
        return wins / len(lines)

    def run(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(l) for l in lines]

        with open(self.output_file, 'w') as f:
            for l in tqdm(lines):
                prompt = l['prompt']
                model_outputs = self.g_eval.get_model_outputs(prompt)
                judge_outputs = self.g_eval.get_judge_outputs(prompt)
                winner = self.g_eval.get_winner(judge_outputs)
                l['model_outputs'] = model_outputs
                l['judge_outputs'] = judge_outputs
                l['models'] = [self.g_eval.model1.name, self.g_eval.model2.name]
                l['judge'] = self.g_eval.judge.model  # model is a string for OpenAI models
                l['winner'] = winner
                f.write(json.dumps(l) + '\n')

        print(f'win rate for model 0: {self.win_rate(0)}')
        print(f'win rate for model 1: {self.win_rate(1)}')


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Run an evaluation with GPT4 judging results.')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model1_url', type=str, default=None)
    parser.add_argument('--model1_name', type=str, default=None)
    parser.add_argument('--judge',
                        type=str,
                        required=True,
                        choices=['gpt4', 'lima', 'vicuna'])
    parser.add_argument('--model2', type=str, default=None)
    parser.add_argument('--model2_url', type=str, default=None)
    parser.add_argument('--model2_name', type=str, default=None)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.model1):
        model1 = CompletionFunction.from_file(args.model1, args.model1_name)
    elif args.model1_url is not None:
        model1 = CompletionFunction.from_url(args.model1_url, args.model1_name)
    else:
        import transformers

        model1 = CompletionFunction.from_pipeline(
            transformers.pipeline('text-generation',
                                  model=args.model1,
                                  tokenizer=args.model1,
                                  device=0), args.model1)

    if args.compare:
        if args.judge == 'gpt4':
            judge = GPT4ComparisonJudge.from_defaults()
        elif args.judge == 'lima':
            judge = LIMACompareJudge.from_defaults()
        elif args.judge == 'vicuna':
            judge = VicunaJudge.from_defaults()
    else:
        if args.judge == 'gpt4':
            judge = GPT4ScoreJudge.from_defaults()
        elif args.judge == 'lima':
            judge = LIMAScoreJudge.from_defaults()
        elif args.judge == 'vicuna':
            # there is only 1 vicuna judge
            judge = VicunaJudge.from_defaults()

    if args.model2 is not None:
        if os.path.exists(args.model2):
            model2 = CompletionFunction.from_file(args.model2, args.model2_name)
        elif args.model2_url is not None:
            model2 = CompletionFunction.from_url(
                url=args.model2_url,
                model=args.model2,
            )
        else:
            import transformers

            model2 = CompletionFunction.from_pipeline(
                transformers.pipeline('text-generation',
                                      model=args.model2,
                                      tokenizer=args.model2,
                                      device=0), args.model2)

        g_eval = GEval(model1, model2, judge)
        EvalCompareRunner(g_eval, args.input_file, args.output_file).run()
    else:
        g_eval = GEval(model1, judge=judge)
        EvalScoreRunner(g_eval, args.input_file, args.output_file).run()
