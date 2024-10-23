# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight check prior to training."""
from argparse import ArgumentParser
from composer.utils import (
    ObjectStoreTransientError,
    dist,
    maybe_create_object_store_from_uri,
    maybe_create_remote_uploader_downloader_from_uri,
    parse_uri,
    retry,
    validate_credentials,
)
from llmfoundry.utils.exceptions import PrivateLinkNotSupportedError

def check_read_from_uri(uri: str) -> None:
    """
    Checks that we have permissions to read from the user's UC bucket.
    """
    if dist.get_global_rank() == 0:
        _, _, path = parse_uri(uri)
        object_store = maybe_create_object_store_from_uri(uri)
        if object_store is not None:
            try:
                def access_object():
                    if path.endswith(".jsonl"):
                        object_store.get_object_size(path)
                    else:
                        objects = object_store.list_objects(prefix=path)
                        if len(objects) == 0:
                            # SDK/UI validation checks that file exists. If we can't access here, likely PL issue
                            raise FileNotFoundError(f"No files found in {path}")
                retry(
                    ObjectStoreTransientError,
                    num_attempts=3,
                )(access_object)()
            except FileNotFoundError as e:
               raise PrivateLinkNotSupportedError(uri) from e

    dist.barrier()

def check_write_to_save_folder(save_folder: str) -> None:
    """
    Checks that we can write a file to the save_folder location specified by the user.
    """
    if dist.get_global_rank() == 0:
        remote_ud = maybe_create_remote_uploader_downloader_from_uri(
                save_folder,
                loggers=[],
            )
        if remote_ud is not None:
            try:
                # replicates the relevant parts of remote_ud.init() without needing composer State
                file_name_to_test = remote_ud._remote_file_name('.credentials_validated_successfully')
                retry(
                    ObjectStoreTransientError,
                    num_attempts=3,
                )(lambda: validate_credentials(remote_ud.remote_backend, file_name_to_test))()
            # MLflow 403 error is automatically wrapped as ObjectStoreTransientError
            except ObjectStoreTransientError as e:
                raise PrivateLinkNotSupportedError(save_folder) from e

    dist.barrier()

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Pre-flight check to surface GPU cluster connectivity issues.",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="Location to write model checkpoints.",
    )
    args = parser.parse_args()
    if args.train_data_path:
        check_read_from_uri(args.train_data_path)
    if args.eval_data_path:
        check_read_from_uri(args.eval_data_path)
    if args.save_folder:
        check_write_to_save_folder(args.save_folder)