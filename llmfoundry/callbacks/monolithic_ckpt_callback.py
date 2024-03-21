# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import tempfile
from pathlib import Path

import torch
from composer.core import Callback, State
from composer.core.state import (fsdp_get_optim_state_dict,
                                 fsdp_state_dict_type_context)
from composer.loggers import Logger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.utils import (dist, format_name_with_dist_and_time, parse_uri,
                            reproducibility)


class MonolithicCheckpointSaver(Callback):
    """Save a monolithic checkpoint every N batches.

    Args:
        save_folder (str): Folder to save checkpoints to (can be a URI)
        batch_interval (int): Number of batches between checkpoints.
        filename (str): Filename to save checkpoints to.
        overwrite (bool): Whether to overwrite previous checkpoints.
        keep_optimizers (bool): Whether to save the optimizer state in the monolithic checkpoint.
    """

    def __init__(self,
                 save_folder: str,
                 batch_interval: int,
                 filename: str = 'ep{epoch}-ba{batch}.pt',
                 overwrite: bool = False,
                 keep_optimizers: bool = False):
        self.backend, self.bucket_name, self.save_dir_format_str = parse_uri(
            save_folder)
        self.filename_format_str = filename
        self.batch_interval = batch_interval
        self.upload_to_object_store = (self.backend != '')
        self.overwrite = overwrite
        self.keep_optimizers = keep_optimizers
        if self.upload_to_object_store:
            self.remote_ud = RemoteUploaderDownloader(
                bucket_uri=f'{self.backend}://{self.bucket_name}')
        else:
            self.remote_ud = None

    def init(self, state: State, logger: Logger) -> None:
        if self.upload_to_object_store and self.remote_ud is not None:
            self.remote_ud.init(state, logger)
            # updated_logger_destinations = [*logger.destinations, new_remote_ud]
            # logger.destinations = tuple(updated_logger_destinations)
            state.callbacks.append(self.remote_ud)

    def batch_checkpoint(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.batch_interval == 0:
            self._save_checkpoint(state, logger)

    def fit_end(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.batch_interval != 0:
            self._save_checkpoint(state, logger)

    def _save_checkpoint(self, state: State, logger: Logger) -> None:
        del logger  # unused

        filename = format_name_with_dist_and_time(self.filename_format_str,
                                                  state.run_name,
                                                  state.timestamp)
        save_dir = format_name_with_dist_and_time(self.save_dir_format_str,
                                                  state.run_name,
                                                  state.timestamp)
        dir_context_mgr = tempfile.TemporaryDirectory(
        ) if self.upload_to_object_store else contextlib.nullcontext(
            enter_result=save_dir)
        with dir_context_mgr as temp_save_dir:
            # pyright doesn't know about enter_result
            assert isinstance(temp_save_dir, str)

            save_path = str(Path(temp_save_dir) / Path(filename))
            dirname = os.path.dirname(save_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            state_dict = {
                'state': state.state_dict(),
                'rng': reproducibility.get_rng_state()
            }
            # Remove sharded model and optimizer state dicts
            state_dict['state'].pop('optimizers')
            state_dict['state'].pop('model')

            # Add in unsharded model params.
            with fsdp_state_dict_type_context(state.model,
                                              state_dict_type='full'):
                state_dict['state']['model'] = state.model.state_dict()

            # Add in unsharded optimizer state dict.
            if self.keep_optimizers:
                optimizer = state.optimizers[0]
                state_dict['state']['optimizers'] = {
                    type(optimizer).__qualname__:
                        fsdp_get_optim_state_dict(state.model,
                                                  optimizer,
                                                  state_dict_type='full')
                }
            if dist.get_global_rank() == 0:
                torch.save(state_dict, save_path)

            if self.upload_to_object_store and self.remote_ud is not None and dist.get_global_rank(
            ) == 0:
                remote_file_name = str(Path(save_dir) / Path(filename))
                self.remote_ud.upload_file(state=state,
                                           remote_file_name=remote_file_name,
                                           file_path=Path(save_path),
                                           overwrite=self.overwrite)
