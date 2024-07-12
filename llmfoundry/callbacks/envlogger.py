# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import socket
from typing import Any, Optional

import git
import pkg_resources
import psutil
import torch
from composer.core import Callback, State
from composer.loggers import Logger


class EnvironmentLoggerCallback(Callback):
    """A callback for logging environment information during model training.

    This callback collects various pieces of information about the training environment,
    including git repository details, package versions, system information, GPU details,
    distributed training setup, NVIDIA driver information, and Docker container details.

    Args:
        workspace_dir (str): The directory containing the workspace. Defaults to '/workspace'.
        log_git (bool): Whether to log git repository information. Defaults to True.
        log_packages (bool): Whether to log package versions. Defaults to True.
        log_nvidia (bool): Whether to log NVIDIA driver information. Defaults to True.
        log_docker (bool): Whether to log Docker container information. Defaults to True.
        log_system (bool): Whether to log system information. Defaults to False.
        log_gpu (bool): Whether to log GPU information. Defaults to False.
        log_distributed (bool): Whether to log distributed training information. Defaults to False.
        packages_to_log (list[str]): A list of package names to log versions for. Defaults to None.

    The collected information is logged as hyperparameters at the start of model fitting.
    """

    PACKAGES_TO_LOG = [
        'llm-foundry',
        'mosaicml',
        'megablocks',
        'grouped-gemm',
        'torch',
        'flash_attn',
        'transformers',
        'datasets',
        'peft',
    ]

    def __init__(
        self,
        workspace_dir: str = '/workspace',
        log_git: bool = True,
        log_nvidia: bool = True,
        log_docker: bool = True,
        log_packages: bool = True,
        log_system: bool = False,
        log_gpu: bool = False,
        log_distributed: bool = False,
        packages_to_log: Optional[list[str]] = None,
    ):
        self.workspace_dir = workspace_dir
        self.log_git = log_git
        self.log_packages = log_packages
        self.log_nvidia = log_nvidia
        self.log_docker = log_docker
        self.log_system = log_system
        self.log_gpu = log_gpu
        self.log_distributed = log_distributed
        self.env_data: dict[str, Any] = {}
        self.packages_to_log = packages_to_log or self.PACKAGES_TO_LOG

    def _get_git_info(self, repo_path: str) -> dict[str, str]:
        repo = git.Repo(repo_path)
        return {
            'commit_hash': repo.head.commit.hexsha,
            'branch': repo.active_branch.name,
        }

    def _get_package_version(self, package_name: str) -> Optional[str]:
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None

    def _get_system_info(self) -> dict[str, Any]:
        return {
            'python_version': platform.python_version(),
            'os': f'{platform.system()} {platform.release()}',
            'hostname': socket.gethostname(),
            'cpu_info': {
                'model': platform.processor(),
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
            },
        }

    def _get_gpu_info(self) -> dict[str, Any]:
        if torch.cuda.is_available():
            return {
                'model': torch.cuda.get_device_name(0),
                'count': torch.cuda.device_count(),
                'memory': {
                    'total': torch.cuda.get_device_properties(0).total_memory,
                    'allocated': torch.cuda.memory_allocated(0),
                },
            }
        return {'available': False}

    def _get_nvidia_info(self) -> dict[str, Any]:
        if torch.cuda.is_available():
            nccl_version = torch.cuda.nccl.version()  # type: ignore
            return {
                'cuda_version':
                    torch.version.cuda,  # type: ignore[attr-defined]
                'cudnn_version': str(torch.backends.cudnn.version(
                )),  # type: ignore[attr-defined]
                'nccl_version': '.'.join(
                    map(str, nccl_version),
                ),
            }

        return {'available': False}

    def _get_distributed_info(self) -> dict[str, Any]:
        return {
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'local_world_size': int(os.environ.get('LOCAL_WORLD_SIZE', 1)),
            'rank': int(os.environ.get('RANK', 0)),
            'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        }

    def _get_docker_info(self) -> dict[str, Any]:
        from mcli import sdk

        run = sdk.get_run(os.environ['RUN_NAME'])
        image, tag = run.image.split(':')
        return {
            'image': image,
            'tag': tag,
        }

    def fit_start(self, state: State, logger: Logger) -> None:
        # Collect environment data
        if self.log_git:
            self.env_data['git_info'] = {
                folder:
                self._get_git_info(os.path.join(self.workspace_dir, folder))
                for folder in os.listdir(self.workspace_dir)
                if os.path.isdir(os.path.join(self.workspace_dir, folder))
            }

        if self.log_packages:
            self.env_data['package_versions'] = {
                pkg: self._get_package_version(pkg)
                for pkg in self.packages_to_log
            }
        if self.log_nvidia:
            self.env_data['nvidia'] = self._get_nvidia_info()

        if self.log_docker:
            self.env_data['docker'] = self._get_docker_info()
        if self.log_system:
            self.env_data['system_info'] = self._get_system_info()

        if self.log_gpu:
            self.env_data['gpu_info'] = self._get_gpu_info()

        if self.log_distributed:
            self.env_data['distributed_info'] = self._get_distributed_info()

        # Log the collected data
        logger.log_hyperparameters({'environment_data': self.env_data})
