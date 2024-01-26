# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""MosaicML LLM Foundry package setup."""

import os
import re

import setuptools
from setuptools import setup

_PACKAGE_NAME = 'llm-foundry'
_PACKAGE_DIR = 'llmfoundry'
_REPO_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_REAL_PATH = os.path.join(_REPO_REAL_PATH, _PACKAGE_DIR)

# Read the repo version
# We can't use `.__version__` from the library since it's not installed yet
with open(os.path.join(_PACKAGE_REAL_PATH, '__init__.py')) as f:
    content = f.read()
# regex: '__version__', whitespace?, '=', whitespace, quote, version, quote
# we put parens around the version so that it becomes elem 1 of the match
expr = re.compile(r"""^__version__\W+=\W+['"]([0-9\.]*)['"]""", re.MULTILINE)
repo_version = expr.findall(content)[0]

# Use repo README for PyPi description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + long_description[
            end + len(end_tag):]

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

install_requires = [
    'git+https://github.com/mosaicml/composer.git',
    'accelerate>=0.25,<0.26',  # for HF inference `device_map`
    'transformers>=4.37,<4.38',
    'mosaicml-streaming>=0.7.2,<0.8',
    'datasets>=2.16,<2.17',
    'fsspec==2023.6.0',  # newer version results in a bug in datasets that duplicates data
    'sentencepiece==0.1.97',
    'einops==0.7.0',
    'omegaconf>=2.2.3,<3',
    'slack-sdk<4',
    'mosaicml-cli>=0.5.27,<1',
    'onnx==1.14.0',
    'onnxruntime==1.15.1',
    'cmake>=3.25.0,<=3.26.3',  # required for triton-pre-mlir below
    # PyPI does not support direct dependencies, so we remove this line before uploading from PyPI
    'triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python',
    'boto3>=1.21.45,<2',
    'huggingface-hub>=0.17.0,<1.0',
    'beautifulsoup4>=4.12.2,<5',  # required for model download utils
    'tenacity>=8.2.3,<9',
]

extra_deps = {}

extra_deps['dev'] = [
    'pre-commit>=3.4.0,<4',
    'pytest>=7.2.1,<8',
    'pytest_codeblocks>=0.16.1,<0.17',
    'pytest-cov>=4,<5',
    'pyright==1.1.256',
    'toml>=0.10.2,<0.11',
    'packaging>=21,<23',
    'hf_transfer==0.1.3',
]

extra_deps['databricks'] = [
    'mosaicml[databricks]>=0.17.1,<0.18',
    'databricks-sql-connector>=3,<4',
    'databricks-connect==14.1.0',
    'lz4>=4,<5',
]

extra_deps['tensorboard'] = [
    'mosaicml[tensorboard]>=0.17.2,<0.18',
]

extra_deps['gpu'] = [
    'flash-attn==1.0.9',
    # PyPI does not support direct dependencies, so we remove this line before uploading from PyPI
    'xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.9#subdirectory=csrc/xentropy',
]
extra_deps['gpu-flash2'] = [
    'flash-attn==2.5.0',
]

extra_deps['turbo'] = [
    'mosaicml-turbo==0.0.8',
]

extra_deps['peft'] = [
    'loralib==0.1.1',  # lora core
    'bitsandbytes==0.39.1',  # 8bit
    # bitsandbytes dependency; TODO: eliminate when incorporated to bitsandbytes
    'scipy>=1.10.0,<=1.11.0',
    # TODO: pin peft when it stabilizes.
    # PyPI does not support direct dependencies, so we remove this line before uploading from PyPI
    'peft==0.4.0',
]

extra_deps['openai'] = [
    'openai==1.3.8',
    'tiktoken==0.4.0',
]
extra_deps['all-cpu'] = set(
    dep for key, deps in extra_deps.items() for dep in deps if 'gpu' not in key)
extra_deps['all'] = set(dep for key, deps in extra_deps.items() for dep in deps
                        if key not in {'gpu-flash2', 'all-cpu'})
extra_deps['all-flash2'] = set(dep for key, deps in extra_deps.items()
                               for dep in deps
                               if key not in {'gpu', 'all', 'all-cpu'})

setup(
    name=_PACKAGE_NAME,
    version=repo_version,
    author='MosaicML',
    author_email='team@mosaicml.com',
    description='LLM Foundry',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mosaicml/llm-foundry/',
    package_data={
        'llmfoundry': ['py.typed'],
    },
    packages=setuptools.find_packages(
        exclude=['.github*', 'mcli*', 'scripts*', 'tests*']),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.9',
)
