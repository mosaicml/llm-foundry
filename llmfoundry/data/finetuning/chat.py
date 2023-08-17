# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List


class ChatFormatter:
    """A class for formatting the chat history.

    Args:
        system: The system prompt. If None, a default ChatML-formatted prompt is used.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.
        turn_joiner: The string to use to join turns. Defaults to a newline.

    Attributes:
        system: The system prompt.
        user: The user prompt.
        assistant: The assistant prompt.
        response_prefix: The response prefix (anything before {} in the assistant format string)
        response_suffix: The response suffix (anything after {} in the assistant format string)
        turn_joiner: The string to use to join turns.
    """

    def __init__(self, system: str = None, user: str = None, assistant: str = None, turn_joiner: str = '\n') -> None:
        self.system = system if system else '<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n'
        self.user = user if user else '<|im_start|>user\n{}<|im_end|>\n'
        self.assistant = assistant if assistant else '<|im_start|>assistant\n{}<|im_end|>\n'
        self.response_prefix = self.assistant.split('{}')[0]
        self.response_suffix = self.assistant.split('{}')[1] if len(self.assistant.split('{}')) > 1 else ''
        self.turn_joiner = turn_joiner

    def parse_structured_history(self, history: List[Dict[str, str]]) -> List[List[str]]:
        """Parses a structured history into a list of lists of strings.
        
        Currently expects Llama2 format, but easily extended to ShareGPT and other similar formats.
        
        Args:
            history: A list of dictionaries. Each dictionary contains two keys: 'role', (user, assistant, or system) and 'content', the message.
            
        Returns:
            The chat history as a list of lists of strings.
        
        Mutates:
            system: The system prompt (if a system prompt is present in the history)
        """
        if len(history) == 0:
            return []
        if history[0]['role'] == 'system':
            if hasattr(self, "system_fmt"):
                self.system = self.system_fmt.format(history[0]['content'])
            else:
                self.system = history[0]['content']
            history = history[1:]
        if history[0]['role'] != 'user':
            raise ValueError('First message must be from user')
        if history[-1]['role'] == 'user':
            history.append({'role': 'assistant', 'content': ''})
        return [[user['content'], assistant['content']] for user, assistant in zip(history[::2], history[1::2])]

    def as_string(self, history: List[List[str]]) -> str:
        """Returns the chat history as a string.
        
        Args:
            history: A list of lists of strings. Each inner list contains two strings: the user input and the assistant response.
            
        Returns:
            The chat history as a string, formatted in the provided syntax.
        """
        if history == []:
            return ''
        if isinstance(history[0], dict):
            history = self.parse_structured_history(history)
        text = self.system + ''.join([
            self.turn_joiner.join([
                self.user.format(item[0]),
                self.assistant.format(item[1]),
            ]) for item in history[:-1]
        ])
        text += self.user.format(history[-1][0])
        text += self.response_prefix
        return text

    def format_response(self, response: str):
        """Formats a response in the provided syntax.

        It assumes that the prompt contained the prefix of the response as is common when training.
        
        Args:
            response: The response to format.
            
        Returns:
            The response, formatted in the provided syntax.
        """
        return response + self.response_suffix


class ChatMLFormatter(ChatFormatter):
    """A class for formatting the chat history in ChatML syntax.

    Args:
        system: The system prompt. This is an unformatted string that will be wrapped in ChatML syntax.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.
    """

    def __init__(self, system: str = None) -> None:
        self.system_fmt = '<|im_start|>system\n{}<|im_end|>\n'
        if system:
            system = self.system_fmt.format(system)
        super().__init__(system)


class Llama2ChatFormatter(ChatFormatter):
    """A class for formatting the chat history in Llama2Chat syntax.
    
    Args:
        system: The system prompt. If None, a default Llama2Chat-formatted prompt is used.
        user: The user prompt. If None, a default Llama2Chat value is used.
        assistant: The assistant prompt. If None, a default Llama2Chat value is used.
    """

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


    def __init__(self, system: str = None) -> None:
        self.system_fmt = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n"
        if not system:
            system = self.DEFAULT_SYSTEM_PROMPT
        system = self.system_fmt.format(system)
        user ="{}"
        assistant = " [/INST] {} </s><s>[INST] "
        super().__init__(system, user, assistant, turn_joiner='')