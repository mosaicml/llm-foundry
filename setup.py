# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""MosaicML LLM Foundry package setup."""

import copy
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
expr = re.compile(
    r"""^__version__\s*=\s*['"]([0-9]+\.[0-9]+\.[0-9]+(?:\.\w+)?)['"]""",
    re.MULTILINE,
)
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
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

install_requires = [
    'mosaicml[libcloud,wandb,oci,gcs,mlflow]>=0.23.4,<0.24',
    'mlflow>=2.13.2,<2.14',
    'accelerate>=0.25,<0.26',  # for HF inference `device_map`
    'transformers>=4.40,<4.41',
    'mosaicml-streaming>=0.7.6,<0.8',
    'torch>=2.3.0,<2.4',
    'datasets>=2.19,<2.20',
    'fsspec==2023.6.0',  # newer version results in a bug in datasets that duplicates data
    'sentencepiece==0.1.97',
    'einops==0.7.0',
    'omegaconf>=2.2.3,<3',
    'slack-sdk<4',
    'mosaicml-cli>=0.6.10,<1',
    'onnx==1.14.0',
    'onnxruntime==1.15.1',
    'boto3>=1.21.45,<2',
    'huggingface-hub>=0.19.0,<0.23',
    'beautifulsoup4>=4.12.2,<5',  # required for model download utils
    'tenacity>=8.2.3,<9',
    'catalogue>=2,<3',
    'typer<1',
]

extra_deps = {}

extra_deps['dev'] = [
    'coverage[toml]==7.4.4',
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
    'mosaicml[databricks]>=0.23.3,<0.24',
    'databricks-sql-connector>=3,<4',
    'databricks-connect==14.1.0',
    'lz4>=4,<5',
]

extra_deps['tensorboard'] = [
    'mosaicml[tensorboard]>=0.23.4,<0.24',
]

# Flash 2 group kept for backwards compatibility
extra_deps['gpu-flash2'] = [
    'flash-attn==2.5.8',
]

extra_deps['gpu'] = copy.deepcopy(extra_deps['gpu-flash2'])

extra_deps['peft'] = [
    'mosaicml[peft]>=0.23.4,<0.24',
]

extra_deps['openai'] = [
    'openai==1.3.8',
    'tiktoken==0.4.0',
]

extra_deps['megablocks'] = [
    'megablocks==0.5.1',
    'grouped-gemm==0.1.4',
]

extra_deps['all-cpu'] = {
    dep for key, deps in extra_deps.items() for dep in deps
    if 'gpu' not in key and 'megablocks' not in key
}
extra_deps['all'] = {
    dep for key, deps in extra_deps.items() for dep in deps
    if key not in {'gpu-flash2', 'all-cpu'}
}
extra_deps['all-flash2'] = {
    dep for key, deps in extra_deps.items() for dep in deps
    if key not in {'gpu', 'all', 'all-cpu'}
}

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
        exclude=['.github*', 'mcli*', 'scripts*', 'tests*'],
    ),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.9',
    entry_points={
        'console_scripts': ['llmfoundry = llmfoundry.cli.cli:app'],
    },
)
