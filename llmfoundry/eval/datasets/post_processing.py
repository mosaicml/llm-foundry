import re
import string
import functools

# r'\n[A-Za-z0-9#`]'

def _normalize_answer(answer: str):
    """Lower text and remove punctuation, articles and extra whitespace.

    Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
    """

    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation +
                        ''.join([u'‘', u'’', u'´', u'`']))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace('_', ' ')

    return white_space_fix(
        remove_articles(handle_punc(lower(
            replace_underscore(answer))))).strip()

def _early_stopping_postprocessing(output="", labels=None, **kwargs):
    if labels is None:
        labels = []
    stopping_criteria = kwargs.get('stopping_criteria', None)
    if not isinstance(stopping_criteria, list):
        raise ValueError(f"stopping_criteria: `{stopping_criteria}` should be of type list!")
    if stopping_criteria is not None and len(stopping_criteria) > 0:
        output = re.split('|'.join(stopping_criteria),
                                output)[0]
    return  output, labels

def _qa_normalization_postprocessing(output="", labels=None, **kwargs):
    if labels is None:
        labels = []
    cleaned_final_answer = _normalize_answer(output)
    cleaned_sample_labels = {
        _normalize_answer(label) for label in labels
    }

    return cleaned_final_answer, cleaned_sample_labels

def _chain_of_thought_postprocessing(output="", labels=None, **kwargs):
    if labels is None:
        labels = []
    cot_delimiter = kwargs.get('cot_delimiter', None)
    if not isinstance(cot_delimiter, str):
        raise ValueError(f"cot_delimiter: `{cot_delimiter}` should be of type string!")
    if cot_delimiter is not None and len(cot_delimiter) > 0:
        output = output.split(cot_delimiter)[-1]
    return output, labels

def _regex_group_postprocessing(output="", labels=None, **kwargs):
    if labels is None:
        labels = []
    regex = kwargs.get('regex', None)
    regex_group = kwargs.get('regex_group', 0)
    
    if regex is not None and len(regex) > 0:
        match = re.search(regex, output)
        if match:
            output = match.group(regex_group)
    return  output, labels

def make_postprocessing_func(name, **kwargs):
    if name ==  'chain_of_thought':
        return functools.partial(_chain_of_thought_postprocessing, **kwargs)
    elif name == 'early_stopping':
        return functools.partial(_early_stopping_postprocessing, **kwargs)
    elif name == 'regex_group_search':
        return functools.partial(_regex_group_postprocessing, **kwargs)
    elif name == 'qa_normalization':
        return functools.partial(_qa_normalization_postprocessing, **kwargs)
