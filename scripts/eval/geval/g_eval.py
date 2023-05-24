# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from typing import Optional, Tuple

import openai
import requests
from tqdm import tqdm
from transformers import pipeline


class CompletionFunction:
    """An abstraction of str->str.

    Works over either URLs, transformer pipelines, or files of input->output.
    """

    def __init__(self, completion_function):
        self.completion_function = completion_function

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
            return cls(lambda prompt: openai.ChatCompletion.create(
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
                temperature=0,
            ).choices[0].message)
        elif 'openai' in url:
            return cls(lambda prompt: openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0,
            ).choices[0].text)
        else:
            print('Using non-openai API, support is limited')
            return cls(
                lambda prompt: requests.post(url, json={
                    'inputs': prompt
                }).json()['completion'])

    @classmethod
    def from_pipeline(cls, pipeline: pipeline):
        return cls(lambda prompt: pipeline(prompt, max_new_tokens=512)[0][
            'generated_text'])

    @classmethod
    def from_file(cls, file: str):
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(l) for l in lines]
        gpt_dict = {}
        for l in lines:
            gpt_dict[l['prompt']] = l['response']

        return cls(lambda prompt: gpt_dict[prompt])


class GPT4Judge(CompletionFunction):

    def build_prompt(self,
                     prompt: str,
                     model_outputs: Tuple,
                     rubric: str = None):
        raise NotImplementedError

    def __call__(self, prompt: str, model_outputs: Tuple, rubric: str = None):
        return self.completion_function(self.build_prompt(
            prompt, model_outputs))

    @classmethod
    def from_defaults(
        cls,
        url: str = 'https://api.openai.com/v1/chat/completions',
        model: str = 'gpt-4',
        system_message:
        str = 'You judge the following response fairly, explaining your reasoning.'
    ):
        return cls.from_url(url, model, system_message)


class GPT4ComparisonJudge(GPT4Judge):

    def build_prompt(self,
                     prompt: str,
                     model_outputs: Tuple[str, str],
                     rubric: str = None):
        if rubric:
            # template in prompt and model_outputs to rubric
            rubric = rubric.replace('{{prompt}}', prompt)
            rubric = rubric.replace('{{model1}}', model_outputs[0])
            rubric = rubric.replace('{{model2}}', model_outputs[1])
            return rubric
        s = 'Help me to fairly judge the following responses to an instruction. Begin by explaining your reasoning. '
        s += "Then, finish by saying either 'The first response is better' or 'The second response is better'.\n\n"
        s += f'The instructions:\n{prompt}\n\n'
        s += f'The first response:\n{model_outputs[0]}\n\n'
        s += f'The second response:\n{model_outputs[1]}\n\n'
        return s


class GPT4RubricJudge(GPT4Judge):

    def build_prompt(self, prompt: str, model_outputs: Tuple[str]):
        s = 'Help me to fairly judge the following response to an instruction. Begin by explaining your reasoning. '
        s += "Then, finish by saying 'OVERALL SCORE: ${score}, replacing ${score} with a number between 0.0 and 10.0'.\n\n"
        s += f'The instructions:\n{prompt}\n\n'
        s += f'The first response:\n{model_outputs[0]}\n\n'
        return s


class GEval:

    def __init__(
        self,
        judge: CompletionFunction,
        model1: CompletionFunction,
        rubric: Optional[str] = None,
        model2: Optional[CompletionFunction] = None,
    ):
        self.judge = judge
        self.model1 = model1
        self.rubric = rubric
        self.model2 = model2

        try:
            self.get_winner('dummy')
            self.score_model = self.get_winner
        except NotImplementedError:
            try:
                self.get_score('dummy')
                self.score_model = self.get_score
            except NotImplementedError:
                self.get_pass_fail('dummy')
                self.score_model = self.get_pass_fail

    def get_model_outputs(self, prompt: str):
        if not self.model2:
            return (self.model1(prompt),)
        else:
            return self.model1(prompt), self.model2(prompt)

    def get_judge_outputs(self, prompt: str):
        return self.judge(prompt, self.get_model_outputs(prompt), self.rubric)

    def get_winner(self, judge_outputs: str):
        raise NotImplementedError

    def get_score(self, judge_outputs: str):
        raise NotImplementedError

    def get_pass_fail(self, judge_outputs: str):
        raise NotImplementedError


class GPT3_5CompareGEval(GEval):

    def __init__(
        self,
        *args,
        judge: CompletionFunction = GPT4ComparisonJudge.from_defaults(),
        model2: CompletionFunction = CompletionFunction.from_url(
            url='https://api.openai.com/v1/completions',
            model='text-davinci-003',
        ),
        **kwargs,
    ):
        super().__init__(*args, judge=judge, model2=model2, **kwargs)

    def get_winner(self, judge_outputs: str):
        if 'the first response is better' in judge_outputs.lower():
            return 0
        elif 'the second response is better' in judge_outputs.lower():
            return 1
        else:
            warnings.warn(f'judge_outputs = {judge_outputs}')


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
                l['winner'] = winner
                f.write(json.dumps(l) + '\n')

        print(f'win rate for model 0: {self.win_rate(0)}')
