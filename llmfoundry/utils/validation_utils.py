import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import OmegaConf as om
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import datasets
from datasets import get_dataset_split_names
from huggingface_hub import dataset_info

from composer.utils import (ObjectStore, maybe_create_object_store_from_uri, parse_uri)
from llmfoundry.utils import build_tokenizer
from llmfoundry.data import ConcatTokensDataset

from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader
from streaming.base.converters import dataframe_to_mds


def create_om_cfg(FT_API_args: Namespace):
    task_type = FT_API_args.task_type

    train_data_path = FT_API_args.train_data_path
    split = 'train'

    if is_hf_dataset_path(FT_API_args.train_data_path):
      train_data_path, split = '/'.join(FT_API_args.train_data_path.split('/')[:2]), FT_API_args.train_data_path.split('/')[-1]

    model = FT_API_args.model
    max_seq_len = FT_API_args.context_length

    common_args = {
        'drop_last': False,
        'num_workers': 2,
        'prefetch_factor': 2,
        'pin_memory': False,
        'persistent_workers': False,
        'timeout': 0
    }
    if task_type == 'INSTRUCTION_FINETUNE':
        cfg = om.create({
            'dataset': {
                'hf_name': train_data_path,
                'split': split,
                'max_seq_len': max_seq_len,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'shuffle': True,
            },
            **common_args
        })

    else:
        cfg = om.create({
            'name': 'finetuning',
            'dataset': {
                'remote': train_data_path,
                'local': train_data_path,
                'split': split,
                'max_seq_len': max_seq_len,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'packing_ratio': None,
                'shuffle': True,
            },
            **common_args
        })

    tokenizer = build_tokenizer(
        tokenizer_name=model,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    return cfg, tokenizer

def token_counts_and_validation(FT_API_args):
    from llmfoundry.data.finetuning import build_finetuning_dataloader

    cfg, tokenizer = create_om_cfg(FT_API_args)

    device_batch_size = 1
    dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
    dataloader = dataspec.dataloader
    token_counting_func = dataspec.get_num_tokens_in_batch

    total_tokens = []
    for batch in dataloader:
        n_batch_tokens = token_counting_func(batch)
        if n_batch_tokens == 0:
            raise ValueError("Empty train sample")
        total_tokens.append(n_batch_tokens)
    return total_tokens


def check_HF_datasets(dataset_names_with_splits: list):
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    for dataset_name_with_split in dataset_names_with_splits:
        dataset_name, split = os.path.split(dataset_name_with_split)
        # make sure we have a dataset and split
        if not dataset_name or not split:
            return False, f"Failed to load Hugging Face dataset {dataset_name_with_split}. Please ensure that you include the split name (e.g. 'mosaicml/dolly_hhrlhf/train')."
        # check user access to the dataset
        try:
            _ = dataset_info(dataset_name)
        except:
            token_warning = ''
            if not token:
                token_warning = ' If this is a private dataset, please set your HUGGING_FACE_HUB_TOKEN using: mcli create secret hf.'
            return False, f"Failed to load Hugging Face dataset {dataset_name_with_split}. Please ensure that the dataset exists and that you have access to it. Remember to include the split name (e.g. 'mosaicml/dolly_hhrlhf/train')." + token_warning
        # check that split exists
        try:
            splits = get_dataset_split_names(dataset_name)
        except:  # error raised in the case of multiple subsets
            return False, f'Failed to load Hugging Face dataset {dataset_name_with_split}. Please make sure that the split is valid and that your dataset does not have subsets.'
        if split not in splits:
            return False, f'Failed to load Hugging Face dataset {dataset_name_with_split}. Split not found.'
    return True, ''


def is_hf_dataset_path(path: str):
    """Check if a given string is a dataset path used by Hugging Face.

    Args:
        path (str): The string to be checked.

    Returns:
        bool: True if the string is a dataset path, False otherwise.
    """
    # Regular expression to match the dataset path pattern
    pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/?(train|validation|test)?/?$'

    return bool(re.match(pattern, path))

def is_uc_delta_table(name: str):
    """name is in the form of catalog.scheme.tablename

       Args:
           name (str): a string folder/file/table path
       Return:
           (bool): True if name is valid UC delta table format
    """
    return '.' in name and '/' not in name and '\\' not in name and len(name.split('.'))==3

def pandas_processing_fn(df: pd.DataFrame,
                         **args: Any) -> Iterable[Dict[str, bytes]]:
    """Tokenize helper function for dataframe_to_mds.

    Args:
        df (pandas.DataFrame): The input pandas DataFrame that needs to be processed.
        **args : Additional arguments to be passed to the 'process_some_data' function during processing.

    Returns:
        iterable obj
    """
    hf_dataset = hf_datasets.Dataset.from_pandas(df=df)
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer'])
    tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace
    dataset = ConcatTokensDataset(
        hf_dataset=hf_dataset,
        max_length=args.get('concat_tokens', None),
        tokenizer=tokenizer,
        eos_text=args.get('eos_text', None),
        bos_text=args.get('bos_text', None),
        no_wrap=args.get('no_wrap', None),
    )

    for sample in dataset:  # pyright: ignore
        yield sample

def integrity_check(out: Union[str, Tuple[str, str]]):
    """Check if the index file has integrity.

       If index is a cloud url, first download it to a temp local file.

    Args:
        out (Union[str, Tuple[str,str]]): MDS dataset path
    """

    def count_shards(mds_root: str):
        n_shard_files = 0
        cu = CloudUploader.get(mds_root, exist_ok=True, keep_local=True)
        for o in cu.list_objects():
            if o.endswith('.mds'):
                n_shard_files += 1
        return n_shard_files

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        if cu.remote:
            download_file(os.path.join(cu.remote, 'index.json'),
                          os.path.join(temp_dir, 'index.json'),
                          timeout=60)
            actual_n_shard_files = count_shards(cu.remote)
            local_merged_index_path = os.path.join(temp_dir, 'index.json')
        else:
            local_merged_index_path = os.path.join(cu.local, 'index.json')
            actual_n_shard_files = count_shards(cu.local)

        merged_index = json.load(open(local_merged_index_path, 'r'))
        n_shard_files = len(
            {b['raw_data']['basename'] for b in merged_index['shards']})
        return n_shard_files == actual_n_shard_files



import logging
import math
import os
import tempfile
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Iterable, List, Tuple, cast

import psutil
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from llmfoundry.data import ConcatTokensDataset
from llmfoundry.utils.data_prep_utils import (DownloadingIterable,
                                              merge_shard_groups)

log = logging.getLogger(__name__)
DONE_FILENAME = '.text_to_mds_conversion_done'


def parse_args( tokenizer,
                concat_tokens,
                output_folder,
                input_folder,
                compression = 'zstd',
                bos_text = '',
                eos_text = '',
                no_wrap = False ,
                processes = 32,
                reprocess = True ) -> Namespace:
    parsed = Namespace(tokenizer = tokenizer,
                       concat_tokens = concat_tokens,
                       output_folder = output_folder,
                       input_folder = input_folder,
                       eos_text = eos_text,
                       bos_text = bos_text,
                       no_wrap = no_wrap,
                       compression = compression,
                       processes = processes,
                       reprocess = reprocess)
    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')
    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def original_parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='The folder to write output to',
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='The folder with text files to convert to mds',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        help='The compression algorithm to use for MDS writing',
    )

    parser.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens',
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        help='The name of the tokenizer to use',
    )
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples',
    )
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=min(max(psutil.cpu_count() - 2, 1), 32),
        help=
        'The number of processes to use to download and convert the dataset',
    )
    parser.add_argument(
        '--reprocess',
        type=bool,
        required=False,
        default=False,
        help='If true, reprocess the input_folder to mds format. Otherwise, ' +
        'only reprocess upon changes to the input folder or dataset creation parameters.',
    )

    parsed = parser.parse_args()

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def get_object_names(input_folder: str) -> List[str]:
    """Get object names from a local or remote folder.

    Args:
        input_folder (str): local or remote folder path.
    """
    object_store = maybe_create_object_store_from_uri(input_folder)
    if object_store is not None:
        _, _, folder_prefix = parse_uri(input_folder)
        names = [
            name for name in object_store.list_objects(folder_prefix)
            if name.endswith('.txt')
        ]
    else:
        # input_folder is a local folder
        names = [
            text_file for dirpath, _, _ in os.walk(input_folder)
            for text_file in glob(os.path.join(dirpath, '*.txt'))
        ]
    # return names, sizes
    log.info(f'Found {len(names)} text files at {input_folder}')

    return names


def get_task_args(
    object_names: List[str],
    output_root: str,
    input_folder: str,
    n_groups: int,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
) -> Iterable:
    """Get download_and_convert arguments split across n_groups.

    Each group handles a portion of object_names.

    Args:
        object_names (List[str]): Names of objects to process
        output_root (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        n_groups (int): Number of groups to split the object names into
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
    """
    num_objects = len(object_names)
    objs_per_group = math.ceil(num_objects / n_groups)
    for group, i in enumerate(range(0, num_objects, objs_per_group)):
        output_subdir = os.path.join(output_root, str(group))
        yield (
            object_names[i:min(i + objs_per_group, num_objects)],
            output_subdir,
            input_folder,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
        )


def download_and_convert_starargs(args: Tuple):
    """Helper function to call download_and_convert with star args.

    This helps us use download_and_convert with mutiprocessing.
    """
    return download_and_convert(*args)


def download_and_convert(
    file_names: List[str],
    output_folder: str,
    input_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
):
    """Downloads and converts text fies to MDS format.

    Args:
        file_names (List[str]): Files to process
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
    """
    object_store = maybe_create_object_store_from_uri(input_folder)

    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloading_iter = DownloadingIterable(object_names=file_names,
                                               output_folder=tmp_dir,
                                               object_store=object_store)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up
        # to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=downloading_iter,
            max_length=concat_tokens,
            tokenizer=tokenizer,
            eos_text=eos_text,
            bos_text=bos_text,
            no_wrap=no_wrap,
        )

        columns = {'tokens': 'bytes'}

        log.info('Converting to MDS format...')
        with MDSWriter(out=output_folder,
                       columns=columns,
                       compression=compression) as out:
            for sample in tqdm(dataset):
                out.write(sample)


def is_remote_path(path: str) -> bool:
    """Checks whether a path is a remote path.

    Args:
        path (str): path to check
    """
    backend, _, _ = parse_uri(path)
    return backend != ''


def is_already_processed(output_root: str, args_str: str,
                         object_names: List[str]) -> bool:
    """Determines whether a group of text files has already been processed.

    Checks the done fie at output root to determine this.

    Args:
        output_root (str): Output folder where a done file may exist
        args_str (str): String representation of the arguments
        object_names (List[str]): Names of objects to convert to MDS format
    """
    # Retrieve the done file contents
    output_object_store = maybe_create_object_store_from_uri(output_root)
    if output_object_store is not None:
        # Download and read the done file from the remote object store
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, DONE_FILENAME)
                output_object_store.download_object(
                    os.path.join(output_folder_prefix, DONE_FILENAME),
                    done_file)
                with open(done_file) as df:
                    done_file_contents = df.read().splitlines()
        except FileNotFoundError:
            return False
    else:
        # Read the local done file
        done_file = os.path.join(output_root, DONE_FILENAME)
        if not os.path.isfile(done_file):
            return False
        with open(done_file) as df:
            done_file_contents = df.read().splitlines()
    # Compare the arguments
    prev_args_str = done_file_contents[0]
    if prev_args_str != args_str:
        return False

    # Compare file names
    prev_names = done_file_contents[1:]
    if len(prev_names) != len(object_names):
        return False
    for idx, prev_name in enumerate(prev_names):
        if object_names[idx] != prev_name:
            return False
    return True


def write_done_file(folder: str, args_str: str, object_names: List[str]):
    """Write a file to signify completion.

    This the done file includes the arguments to processing and
    a list of objects that were processed.

    Args:
        folder (str): Folder to write the done file to
        args_str (str): String representation of arguments
        object_names (List[str]): List of objects to convert to MDS format
    """
    with open(os.path.join(folder, DONE_FILENAME), 'w') as done_file:
        done_file.write('\n'.join([args_str] + object_names) + '\n')


def convert_text_to_mds(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    processes: int,
    args_str: str,
    reprocess: bool,
):
    """Convert a folder of text files to MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        processes (int): The number of processes to use.
        args_str (str): String representation of the arguments
        reprocess (bool): Whether to always reprocess the given folder of text files
    """
    is_remote_output = is_remote_path(output_folder)

    object_names = get_object_names(input_folder)
    if len(object_names) == 0:
        raise ValueError(f'No text files were found at {input_folder}.')

    # Check if the text files in the bucket have already been processed.
    if not reprocess and is_already_processed(output_folder, args_str,
                                              object_names):
        log.info(
            f'Input folder {input_folder} is already processed at {output_folder} and '
            +
            'reprocess is set to False. Set reprocess to True if you would like to force reprocessing.'
        )
        return

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory(
    ).name if is_remote_output else output_folder

    if processes > 1:
        # Download and convert the text files in parallel
        args = get_task_args(object_names, local_output_folder, input_folder,
                             processes, tokenizer_name, concat_tokens, eos_text,
                             bos_text, no_wrap, compression)
        with ProcessPoolExecutor(max_workers=processes) as executor:
            list(executor.map(download_and_convert_starargs, args))

        # Merge the mds shards from each of the processes into a single folder
        merge_shard_groups(local_output_folder)
    else:
        download_and_convert(object_names, local_output_folder, input_folder,
                             tokenizer_name, concat_tokens, eos_text, bos_text,
                             no_wrap, compression)

    # Write a done file with the args and object names
    write_done_file(local_output_folder, args_str, object_names)

    if is_remote_output:
        # Upload the local output to the remote location
        output_object_store = cast(
            ObjectStore, maybe_create_object_store_from_uri(output_folder))
        _, _, output_folder_prefix = parse_uri(output_folder)
        files_to_upload = os.listdir(local_output_folder)

        for file in files_to_upload:
            assert not os.path.isdir(file)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(
                remote_path, os.path.join(local_output_folder, file))


def _args_str(original_args: Namespace) -> str:
    """Create a string from the args to determine whether to reprocess.

    Args:
        original_args (Namespace): Arguments to main function.
    """
    # Take the arguments that influence the final result.
    # reprocess and max_mds_writer_workers are not taken.
    args = Namespace(
        tokenizer_name=original_args.tokenizer,
        output_folder=original_args.output_folder,
        input_folder=original_args.input_folder,
        concat_tokens=original_args.concat_tokens,
        eos_text=original_args.eos_text,
        bos_text=original_args.bos_text,
        no_wrap=original_args.no_wrap,
        compression=original_args.compression,
        processes=original_args.processes,
    )

    return str(args)



from composer.utils import dist, get_file, parse_uri
from llmfoundry.data.finetuning.tasks import (DOWNLOADED_FT_DATASETS_DIRPATH,
                                              SUPPORTED_EXTENSIONS,
                                              dataset_constructor)

def _download_remote_hf_dataset(remote_path: str, split: str) -> str:
    """Downloads a dataset from a remote object store.

    This function supports 'jsonl', 'csv', and 'parquet' file formats for the dataset. It will attempt to download
    the dataset, then once it is downloaded, convert it into HuggingFace ``datasets`` format, and then return this
    dataset.

    The function also ensures synchronicity across multiple processes during the file download. It creates a signal
    file that is used to synchronize the start of the download across different processes. Once the download is
    completed, the function removes the signal file.

    Args:
        hf_name (str): The path of the HuggingFace dataset to download.
        split (str): The dataset split to download (e.g., 'train', 'validation', 'test').

    Returns:
        A local directory path where the dataset files are stored.

    Raises:
        FileNotFoundError: Raised if the dataset file cannot be found with any of the supported extensions.
    """
    finetune_dir = os.path.join(
        DOWNLOADED_FT_DATASETS_DIRPATH,
        split if split != 'data' else 'data_not',
    )
    os.makedirs(finetune_dir, exist_ok=True)
    for extension in SUPPORTED_EXTENSIONS:
        name = f'{remote_path.strip("/")}/{split}{extension}'
        destination = str(
            os.path.abspath(
                os.path.join(finetune_dir, 'data',
                             f'{split}-00000-of-00001{extension}')))

        # Since we don't know exactly what the extension will be, since it is one of a list
        # use a signal file to wait for instead of the desired file
        signal_file_path = os.path.join(
            finetune_dir, f'.node_{dist.get_node_rank()}_local_rank0_completed')
        if dist.get_local_rank() == 0:
            try:
                get_file(path=name, destination=destination, overwrite=True)
            except FileNotFoundError as e:
                if extension == SUPPORTED_EXTENSIONS[-1]:
                    files_searched = [
                        f'{cfg.dataset.hf_name}/{cfg.dataset.split}{ext}'
                        for ext in SUPPORTED_EXTENSIONS
                    ]
                    raise FileNotFoundError(
                        f'Could not find a file with any of ' + \
                        f'the supported extensions: {SUPPORTED_EXTENSIONS}\n' + \
                        f'at {files_searched}'
                    ) from e
                else:
                    log.debug(
                        f'Could not find {name}, looking for another extension')
                continue

            os.makedirs(os.path.dirname(signal_file_path), exist_ok=True)
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_download')

        # Avoid the collective call until the local rank zero has finished trying to download the dataset
        # so that we don't timeout for large downloads. This syncs all processes on the node
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            # Then, wait to ensure every node has finished trying to download the dataset
            dist.barrier()

        # clean up signal file
        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)
        dist.barrier()
        break
    return finetune_dir

