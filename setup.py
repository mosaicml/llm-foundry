"""MosaicML examples package setup."""

import os
import re
from typing import Dict, List

import setuptools
from setuptools import find_packages, setup

_PACKAGE_NAME = 'mosaicml-examples'
_PACKAGE_DIR = 'examples'
_EXAMPLE_SUBDIRS = ('cifar', 'resnet', 'deeplab', 'bert', 'llm')
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
]

with open(os.path.join(_REPO_REAL_PATH, 'requirements.txt'), 'r') as f:
    install_requires = f.readlines()


def _dependencies_as_dict(deps: List[str]) -> Dict[str, str]:
    """map, e.g., 'foo>=1.5,<1.6' -> {'foo': '>=1.5,<1.6'}"""
    ret = {}
    for dep in deps:
        elems = re.split('([=><])', dep.strip())
        ret[elems[0]] = ''.join(elems[1:])
    return ret


def _merge_dependencies(deps_base: List[str],
                        deps_update: List[str],
                        cpu_only: bool = False):
    """Subdir requirements.txt supersedes repo requirements."""
    base_dict = _dependencies_as_dict(deps_base)
    base_dict.update(_dependencies_as_dict(deps_update))
    base_dict.pop(_PACKAGE_NAME, None)  # avoid circular dependencies
    if cpu_only:
        # these packages can't even be installed unless there's actually
        # a GPU on your machine
        base_dict.pop('flash-attn', None)
        base_dict.pop('triton', None)
    return [k + v for k, v in base_dict.items()]  # 'foo': '>3' -> 'foo>3'


extra_deps = {}
for name in _EXAMPLE_SUBDIRS:
    subdir_path = os.path.join(_PACKAGE_REAL_PATH, name, 'requirements.txt')
    with open(subdir_path, 'r') as f:
        lines = f.readlines()
        extra_deps[name] = _merge_dependencies(install_requires,
                                               lines,
                                               cpu_only=False)
        extra_deps[f'{name}-cpu'] = _merge_dependencies(install_requires,
                                                        lines,
                                                        cpu_only=True)

setup(
    name=_PACKAGE_NAME,
    version=repo_version,
    author='MosaicML',
    author_email='team@mosaicml.com',
    description='Optimized starter code for deep learning training + evaulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mosaicml/examples/',
    package_dir={_PACKAGE_DIR: _PACKAGE_REAL_PATH},
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.7',
)
