# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Databricks notebook source
# MAGIC %md
# MAGIC JIRA: https://databricks.atlassian.net/jira/software/c/projects/STR/issues/STR-141?filter=allissues

# COMMAND ----------

# MAGIC %md
# MAGIC ## Warning: Important Alert Regarding the Script Usage
# MAGIC
# MAGIC ### Script Purpose:
# MAGIC - **Not for Training**: This script is not utilized during the training process.
# MAGIC - **Ad-Hoc Validation**: It serves as an ad-hoc utility for users to run independently prior to starting fine-tuning.
# MAGIC - **Data Verification**: Its primary function is to validate the user's data before they invoke the Fine-Tuning (FT) API.
# MAGIC - **Cost Estimation**: Users can estimate the cost implications with this script.
# MAGIC
# MAGIC ### Usage Scenario:
# MAGIC This script is particularly useful in scenarios where there is a risk of data being malformed. It acts as a preventive measure to ensure data integrity and helps in cost assessment for the fine-tuning process.
# MAGIC
# MAGIC ### Note on Long-Term Solution:
# MAGIC - **Temporary Measure**: This script is a stop-gap solution.
# MAGIC - **Future Development**: We are in the process of developing a long-term data preparation service, which will eventually replace this script.
# MAGIC
# MAGIC ### Checks Include:
# MAGIC - check input dataset:
# MAGIC   1) verify if dataset input format is valid (need to be one of these: Huggingface, delta table, dbfs:/Volumes, cloud path);
# MAGIC - check HF input location:
# MAGIC   1) load dataset info and check if it is accessible;
# MAGIC   2) verify if the split exists.
# MAGIC - check cloud path location:
# MAGIC   1) check the cloud prefix is compliant with composers' object store supports (gs, s3, oci)
# MAGIC   2) check if list objects returns nothing.
# MAGIC - count_tokens:
# MAGIC   1) For IFT task: validate tokenization by running tokenizer + filter on the entire dataset. count the number of tokens. Throws error if there are any empty responses or prompts
# MAGIC   2) For CPT task: call donwload_text_to_mds.py and count the resulted mds dataset. Note this could take a long time.
# MAGIC
# MAGIC ### Questions:
# MAGIC - Is "download_text_to_mds.py" always callable from the validation script?
# MAGIC - what is the function to reuse to run tokenization on HF datasets with filters?
# MAGIC - The inputs to this validation script is assumed to be the same or a subset of the FT API arguments, i.e., a configuration like below. Is this a valid assumption?
# MAGIC ```
# MAGIC cfg = {
# MAGIC     model: str,
# MAGIC     train_data_path: str,
# MAGIC     save_folder: str,
# MAGIC     *,
# MAGIC     task_type: Optional[str] = "INSTRUCTION_FINETUNE",
# MAGIC     eval_data_path: Optional[str] = None,
# MAGIC     eval_prompts: Optional[List[str]] = None,
# MAGIC     custom_weights_path: Optional[str] = None,
# MAGIC     training_duration: Optional[str] = None,
# MAGIC     learning_rate: Optional[float] = None,
# MAGIC     context_length: Optional[int] = None,
# MAGIC     experiment_trackers: Optional[List[Dict]] = None,
# MAGIC     data_prep_config: Optional[Dict] = None,
# MAGIC     disable_credentials_check: Optional[bool] = None,
# MAGIC     timeout: Optional[float] = 10,
# MAGIC     future: Literal[False] = False,
# MAGIC }
# MAGIC - What null checkings do we want to have?
# MAGIC - How to map the model to its expected eos_text / bos_text format? [Ref](https://databricks.slack.com/archives/C05K29T9NBF/p1703644153357929?thread_ts=1703643155.904289&cid=C05K29T9NBF)
# MAGIC - How to automate tokenization for CPT? it is always really standard: sequence -> concat(tok(BOS), tok(sequence), tok(EOS)), and then concatenate sequences. [Ref](https://databricks.slack.com/archives/C05K29T9NBF/p1703698056000399?thread_ts=1703643155.904289&cid=C05K29T9NBF)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install llm-foundry

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import os
import re
from argparse import ArgumentParser, Namespace
from typing import Tuple, Union

from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from datasets import get_dataset_split_names
from huggingface_hub import dataset_info
from omegaconf import OmegaConf as om

from llmfoundry.utils import build_tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC ## User Defines the Cell Below

# COMMAND ----------

FT_API_args = Namespace(
    model='EleutherAI/gpt-neox-20b',
    train_data_path=
    'tatsu-lab/alpaca',  # 'mosaicml/dolly_hhrlhf/train', # tatsu-lab/alpaca/train',
    save_folder=
    'dbfs:/databricks/mlflow-tracking/EXPERIMENT_ID/RUN_ID/artifacts/checkpoints',
    task_type='INSTRUCTION_FINETUNE',
    eval_data_path=None,
    eval_prompts=None,
    custom_weights_path=None,
    training_duration=None,
    learning_rate=None,
    context_length=2048,
    experiment_trackers=None,
    disable_credentials_check=None,
    # Extra argument to add to FT API
    # See comment https://databricks.atlassian.net/browse/STR-141?focusedCommentId=4308948
    data_prep_config={
        'data_validation': True,
        'data_prep': False
    },
    timeout=10,
    future=False,
)

os.environ['HF_ASSETS_CACHE'] = '/tmp/'
os.environ['HF_HOME'] = '/tmp/'
os.environ['HF_HUB_CACHE'] = '/tmp/'
os.environ['HF_DATASETS_CACHE'] = '/tmp/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adapted from llmfoundry/scripts/data_prep/convert_text_to_mds.py

# COMMAND ----------

# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Taken from llmfoundry/scripts/data_prep/convert_text_to_mds.py

import logging
import math
import tempfile
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Iterable, List, Tuple, cast

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


def parse_args(
    tokenizer: str,
    concat_tokens: int,
    output_folder: str,
    input_folder: str,
    compression: str = 'zstd',
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    processes: int = 32,  # min(max(psutil.cpu_count() - 2, 1), 32),
    reprocess: bool = False
) -> Namespace:

    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing',
    )
    parsed = Namespace(tokenizer=tokenizer,
                       concat_tokens=concat_tokens,
                       output_folder=output_folder,
                       input_folder=input_folder,
                       eos_text=eos_text,
                       bos_text=bos_text,
                       no_wrap=no_wrap,
                       compression=compression,
                       processes=processes,
                       reprocess=reprocess)

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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Inputs and Count tokens

# COMMAND ----------

import json

from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader


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


def create_om_cfg(FT_API_args: Namespace):
    task_type = FT_API_args.task_type
    train_data_path = FT_API_args.train_data_path
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
                'split': 'train',
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
                'split': 'train',
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


# COMMAND ----------


# build cfg from the inputs
def main():
    if FT_API_args.task_type == 'INSTRUCTION_FINETUNE':
        # check if train_data_path is a valid HF dataset url with splits.
        if not is_hf_dataset_path(FT_API_args.train_data_path):
            raise ValueError(
                f'Input path {FT_API_args.train_data_path} is not supported. It needs to be a valid Huggingface dataset path.'
            )
        # load dataset.info and see if HF tokens are correctly set.
        check_HF_datasets(FT_API_args.train_data_path)

        cfg, tokenizer = create_om_cfg(FT_API_args)

    elif FT_API_args.task_type == 'CONTINUED_PRETRAIN':
        # check if train_data_path is a valid object store that composer supports
        cfg, tokenizer = create_om_cfg(FT_API_args)

        input_folder = FT_API_args.train_data_path
        output_folder = FT_API_args.save_folder
        concat_tokens = FT_API_args.context_length
        tokenizer_name = FT_API_args.model

        # Run convert_text_to_mds.py and dump MDS dataset to "save_folder"
        args = parse_args(tokenizer, concat_tokens, output_folder, input_folder)
        convert_text_to_mds(tokenizer_name=args.tokenizer,
                            output_folder=args.output_folder,
                            input_folder=args.input_folder,
                            concat_tokens=args.concat_tokens,
                            eos_text=args.eos_text,
                            bos_text=args.bos_text,
                            no_wrap=args.no_wrap,
                            compression=args.compression,
                            processes=args.processes,
                            reprocess=args.reprocess,
                            args_str=_args_str(args))

        # Check if the MDS dataset is integral by checking index.json
        if integrity_check(args.output_folder):
            raise RuntimeError(
                f'{args.output_folder} has mismatched number of shard files between merged index.json and actual shards!'
            )

        print('Converted data for continnued pre-training was saved in: ',
              args.output_folder)

    else:
        raise ValueError(
            f'task_type can only be INSTRUCTION_FINETUNE or Continued_Pretraining but got {FT_API_args.task_type} instead!'
        )
    # Run a few checks on resulted MDS datasets
    # 1. no shards in output_folder
    # 2. check shard completeness by downloading and inspecting index.json

    from llmfoundry.data.finetuning import build_finetuning_dataloader
    tokenizer_name = 'EleutherAI/gpt-neox-20b'
    tokenizer_kwargs = {'model_max_length': cfg.dataset.max_seq_len}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    device_batch_size = 1
    dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
    dataloader = dataspec.dataloader
    token_counting_func = dataspec.get_num_tokens_in_batch

    total_tokens = 0
    for batch in dataloader:
        total_tokens += token_counting_func(batch)

    print('Total number of tokens:', total_tokens)


# COMMAND ----------

if __name__ == '__main__':
    main()
