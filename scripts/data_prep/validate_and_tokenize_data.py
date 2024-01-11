# Databricks notebook source
# MAGIC %md
# MAGIC Copyright 2022 MosaicML LLM Foundry authors.
# MAGIC SPDX-License-Identifier: Apache-2.0

# COMMAND ----------

# MAGIC %md
# MAGIC JIRA: https://databricks.atlassian.net/jira/software/c/projects/STR/issues/STR-141?filter=allissues

# COMMAND ----------

# MAGIC %md
# MAGIC # Warning: Important Alert Regarding the Script Usage
# MAGIC
# MAGIC #### Usage Scenario:
# MAGIC This script is particularly designed for Databricks' customers who have access to Databricks notebook and UC. Our customers may find this script useful in scenarios where there is a risk of data being malformed. It acts as a preventive measure to ensure data integrity and helps in cost assessment for the fine-tuning process.
# MAGIC
# MAGIC #### Script Purpose:
# MAGIC - **Not for Training**: This script is not utilized during the training process.
# MAGIC - **Ad-Hoc Validation**: It serves as an ad-hoc utility for users to run independently prior to starting fine-tuning.
# MAGIC - **Data Verification**: Its primary function is to validate the user's data before they invoke the Fine-Tuning (FT) API.
# MAGIC - **Cost Estimation**: Users can estimate the cost implications with this script.
# MAGIC
# MAGIC #### Note on Long-Term Solution:
# MAGIC - **Temporary Measure**: This script is a stop-gap solution.
# MAGIC - **Future Development**: We are in the process of developing a long-term data preparation service, which will eventually replace this script.
# MAGIC
# MAGIC #### User Defines:
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
# MAGIC ``` 
# MAGIC
# MAGIC #### Checks Include:
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
# MAGIC #### To-dos: 
# MAGIC - Map the model to its expected eos_text / bos_text format automatically [Ref](https://databricks.slack.com/archives/C05K29T9NBF/p1703644153357929?thread_ts=1703643155.904289&cid=C05K29T9NBF)
# MAGIC - Automate tokenization for CPT. it is always really standard: sequence -> concat(tok(BOS), tok(sequence), tok(EOS)), and then concatenate sequences. [Ref](https://databricks.slack.com/archives/C05K29T9NBF/p1703698056000399?thread_ts=1703643155.904289&cid=C05K29T9NBF)
# MAGIC - Add ``preprocessing_fn`` here. -- We don't need to. FT API does not expose preprocessing_fn. 
# MAGIC - Add a sample_ratio parameter so users can run the validation on a portion of the whole dataest then estimate by the scaling factor. 
# MAGIC - Put utility function in a validation branch. 
# MAGIC - 

# COMMAND ----------

# %pip install git+https://github.com/mosaicml/llm-foundry.git@byod/data_validation
%pip install git+https://github.com/XiaohanZhangCMU/llm-foundryX.git@validation 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Instruction Fine Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC #### All Utility Functions

# COMMAND ----------

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
from llmfoundry.utils import (create_om_cfg, token_counts_and_validation,
        check_HF_datasets, is_hf_dataset_path, is_uc_delta_table,
        pandas_processing_fn, integrity_check, convert_text_to_mds,
        _args_str)
from llmfoundry.data import ConcatTokensDataset

from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader
from streaming.base.converters import dataframe_to_mds

  
# def create_om_cfg(FT_API_args: Namespace):
#     task_type = FT_API_args.task_type

#     train_data_path = FT_API_args.train_data_path
#     split = 'train'

#     if is_hf_dataset_path(FT_API_args.train_data_path):
#       train_data_path, split = '/'.join(FT_API_args.train_data_path.split('/')[:2]), FT_API_args.train_data_path.split('/')[-1] 

#     model = FT_API_args.model
#     max_seq_len = FT_API_args.context_length

#     common_args = {
#         'drop_last': False,
#         'num_workers': 2,
#         'prefetch_factor': 2,
#         'pin_memory': False,
#         'persistent_workers': False,
#         'timeout': 0
#     }
#     if task_type == 'INSTRUCTION_FINETUNE':
#         cfg = om.create({
#             'dataset': {
#                 'hf_name': train_data_path,
#                 'split': split,
#                 'max_seq_len': max_seq_len,
#                 'decoder_only_format': True,
#                 'allow_pad_trimming': False,
#                 'shuffle': True,
#             },
#             **common_args
#         })

#     else:
#         cfg = om.create({
#             'name': 'finetuning',
#             'dataset': {
#                 'remote': train_data_path,
#                 'local': train_data_path,
#                 'split': split,
#                 'max_seq_len': max_seq_len,
#                 'decoder_only_format': True,
#                 'allow_pad_trimming': False,
#                 'packing_ratio': None,
#                 'shuffle': True,
#             },
#             **common_args
#         })

#     tokenizer = build_tokenizer(
#         tokenizer_name=model,
#         tokenizer_kwargs={'model_max_length': max_seq_len},
#     )

#     return cfg, tokenizer

# def token_counts_and_validation(FT_API_args):
#     from llmfoundry.data.finetuning import build_finetuning_dataloader

#     cfg, tokenizer = create_om_cfg(FT_API_args) 

#     device_batch_size = 1
#     dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
#     dataloader = dataspec.dataloader
#     token_counting_func = dataspec.get_num_tokens_in_batch

#     total_tokens = []
#     for batch in dataloader:
#         n_batch_tokens = token_counting_func(batch)
#         if n_batch_tokens == 0: 
#             raise ValueError("Empty train sample")
#         total_tokens.append(n_batch_tokens)
#     return total_tokens
  
# #----------------------------------------   IFT  ---------------------------------------- # 

# def check_HF_datasets(dataset_names_with_splits: list):
#     token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
#     for dataset_name_with_split in dataset_names_with_splits:
#         dataset_name, split = os.path.split(dataset_name_with_split)
#         # make sure we have a dataset and split
#         if not dataset_name or not split:
#             return False, f"Failed to load Hugging Face dataset {dataset_name_with_split}. Please ensure that you include the split name (e.g. 'mosaicml/dolly_hhrlhf/train')."
#         # check user access to the dataset
#         try:
#             _ = dataset_info(dataset_name)
#         except:
#             token_warning = ''
#             if not token:
#                 token_warning = ' If this is a private dataset, please set your HUGGING_FACE_HUB_TOKEN using: mcli create secret hf.'
#             return False, f"Failed to load Hugging Face dataset {dataset_name_with_split}. Please ensure that the dataset exists and that you have access to it. Remember to include the split name (e.g. 'mosaicml/dolly_hhrlhf/train')." + token_warning
#         # check that split exists
#         try:
#             splits = get_dataset_split_names(dataset_name)
#         except:  # error raised in the case of multiple subsets
#             return False, f'Failed to load Hugging Face dataset {dataset_name_with_split}. Please make sure that the split is valid and that your dataset does not have subsets.'
#         if split not in splits:
#             return False, f'Failed to load Hugging Face dataset {dataset_name_with_split}. Split not found.'
#     return True, ''


# def is_hf_dataset_path(path: str):
#     """Check if a given string is a dataset path used by Hugging Face.

#     Args:
#         path (str): The string to be checked.

#     Returns:
#         bool: True if the string is a dataset path, False otherwise.
#     """
#     # Regular expression to match the dataset path pattern
#     pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/?(train|validation|test)?/?$'

#     return bool(re.match(pattern, path))
  
# def is_uc_delta_table(name: str):
#     """name is in the form of catalog.scheme.tablename

#        Args:
#            name (str): a string folder/file/table path
#        Return:
#            (bool): True if name is valid UC delta table format
#     """
#     return '.' in name and '/' not in name and '\\' not in name and len(name.split('.'))==3
    
# #----------------------------------------   CPT  ---------------------------------------- # 

# def pandas_processing_fn(df: pd.DataFrame,
#                          **args: Any) -> Iterable[Dict[str, bytes]]:
#     """Tokenize helper function for dataframe_to_mds.

#     Args:
#         df (pandas.DataFrame): The input pandas DataFrame that needs to be processed.
#         **args : Additional arguments to be passed to the 'process_some_data' function during processing.

#     Returns:
#         iterable obj
#     """
#     hf_dataset = hf_datasets.Dataset.from_pandas(df=df)
#     tokenizer = AutoTokenizer.from_pretrained(args['tokenizer'])
#     tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace
#     dataset = ConcatTokensDataset(
#         hf_dataset=hf_dataset,
#         max_length=args.get('concat_tokens', None),
#         tokenizer=tokenizer,
#         eos_text=args.get('eos_text', None),
#         bos_text=args.get('bos_text', None),
#         no_wrap=args.get('no_wrap', None),
#     )

#     for sample in dataset:  # pyright: ignore
#         yield sample

# def integrity_check(out: Union[str, Tuple[str, str]]):
#     """Check if the index file has integrity.

#        If index is a cloud url, first download it to a temp local file.

#     Args:
#         out (Union[str, Tuple[str,str]]): MDS dataset path
#     """

#     def count_shards(mds_root: str):
#         n_shard_files = 0
#         cu = CloudUploader.get(mds_root, exist_ok=True, keep_local=True)
#         for o in cu.list_objects():
#             if o.endswith('.mds'):
#                 n_shard_files += 1
#         return n_shard_files

#     cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         if cu.remote:
#             download_file(os.path.join(cu.remote, 'index.json'),
#                           os.path.join(temp_dir, 'index.json'),
#                           timeout=60)
#             actual_n_shard_files = count_shards(cu.remote)
#             local_merged_index_path = os.path.join(temp_dir, 'index.json')
#         else:
#             local_merged_index_path = os.path.join(cu.local, 'index.json')
#             actual_n_shard_files = count_shards(cu.local)

#         merged_index = json.load(open(local_merged_index_path, 'r'))
#         n_shard_files = len(
#             {b['raw_data']['basename'] for b in merged_index['shards']})
#         return n_shard_files == actual_n_shard_files


# COMMAND ----------

# MAGIC %md 
# MAGIC #### User Defines
# MAGIC Use the same input arguments you will want to provide to FT API

# COMMAND ----------

FT_API_args = Namespace(
    model='EleutherAI/gpt-neox-20b',
    train_data_path= 'tatsu-lab/alpaca/train', # 'main.streaming.random_large_table',  #   # 'mosaicml/dolly_hhrlhf/train', # tatsu-lab/alpaca/train',
    save_folder= 'dbfs:/databricks/mlflow-tracking/EXPERIMENT_ID/RUN_ID/artifacts/checkpoints',
    task_type='INSTRUCTION_FINETUNE',
    training_duration=3,
    context_length=2048,
)

temporary_jsonl_data_path = '/tmp/ft_data/train/'
os.environ['HF_ASSETS_CACHE'] = '/tmp/'
os.environ['HF_HOME'] = '/tmp/'
os.environ['HF_HUB_CACHE'] = '/tmp/'
os.environ['HF_DATASETS_CACHE'] = '/tmp/'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Loading
# MAGIC
# MAGIC The IFT data needs to stay with a format 
# MAGIC ```
# MAGIC prompt: xxx
# MAGIC response or completion: yyy
# MAGIC ```
# MAGIC
# MAGIC Based on FT_API_args.train_data_path, we will select an ingestion method from three options.
# MAGIC
# MAGIC - Option-1. Your data is a JSONL file which stores in an object store supported by Composer. [Example file to-be-added](todo - add a link to such a file)
# MAGIC - Option-2. You provide a Huggingface dataset ID. Note you need to provide a split as well. [Example dataset link to-be-added](huggingface.co)
# MAGIC - Option-3. You have a delta table. 

# COMMAND ----------

raw_dataset = None

if FT_API_args.train_data_path.endswith('.jsonl') and os.path.exists(FT_API_args.train_data_path): 
    data_path = FT_API_args.train_data_path 
    raw_dataset = datasets.load_dataset('json', data_path) 

if is_hf_dataset_path(FT_API_args.train_data_path):
    check_HF_datasets(FT_API_args.train_data_path)
    dataset_id, split = '/'.join(FT_API_args.train_data_path.split('/')[:2]), FT_API_args.train_data_path.split('/')[-1]    
    raw_dataset = datasets.load_dataset(dataset_id, split=split)       

if is_uc_delta_table(FT_API_args.train_data_path):    
    delta_table_name = FT_API_args.train_data_path
    df = spark.read.table(delta_table_name)
    df = df.toPandas()
    df.rename(columns={'prompts': 'prompt', 'responses': 'response'}, inplace=True)
    df.to_json(os.path.join(temporary_jsonl_data_path, 'ift.jsonl'), orient='records', lines=True)    
    raw_dataset = datasets.Dataset.from_pandas(df) 
    FT_API_args.train_data_path = temporary_jsonl_data_path

if raw_dataset is None: 
    raise RuntimeError("Can't find a proper ingestion method")

# COMMAND ----------

!mkdir -p {temporary_jsonl_data_path}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validation and Statistics

# COMMAND ----------

# Initial dataset stats
print("Num examples:", len(raw_dataset))
print("First example:")
for ex in raw_dataset: 
    print(ex)
    print() 
    break 

format_errors = defaultdict(int)

for ex in raw_dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1 
        continue 
      
    prompts = ex.get("prompt", None)
    if not prompts:
        format_errors["missing_prompt"] += 1
        continue

    responses = ex.get("response", None)
    if not responses:
        format_errors["missing_response"] += 1
        continue

if format_errors:
    print("Oops! Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("Congratulations! No errors found")    

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cost Estimation
# MAGIC
# MAGIC Tokenize the raw dataset and we see some statistics of the tokens and estimate the overall cost based on default trainining duration

# COMMAND ----------

MAX_TOKENS_PER_EXAMPLE = FT_API_args.context_length if FT_API_args.context_length is not None else 4096
TARGET_EPOCHS = FT_API_args.training_duration if FT_API_args.training_duration is not None else 1 
n_epochs = TARGET_EPOCHS
n_train_examples = len(raw_dataset)

batch_tokens = token_counts_and_validation(FT_API_args)
n_billing_tokens_in_dataset = sum(batch_tokens)

print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

# COMMAND ----------

# MAGIC %md 
# MAGIC # Continued Pretrain

# COMMAND ----------

# MAGIC %md 
# MAGIC #### User Defines

# COMMAND ----------

FT_API_args = Namespace(
    model='EleutherAI/gpt-neox-20b',
    train_data_path= 'dbfs:/xiaohan-test/test_cpt/', 
    save_folder= 'dbfs:/databricks/mlflow-tracking/EXPERIMENT_ID/RUN_ID/artifacts/checkpoints',
    task_type='CONTINUED_PRETRAIN',
    training_duration=3,
    context_length=2048,
)

temporary_mds_output_path = '/tmp/xiaohan-test/test_mds'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Loading (from text to MDS)
# MAGIC
# MAGIC Copy [llmfoundry/scripts/data_prep/convert_text_to_mds.py](https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/convert_text_to_mds.py) here and run the cell below

# COMMAND ----------

from convert_text_to_mds import convert_text_to_mds, parse_args, _args_str

# check if train_data_path is a valid object store that composer supports
cfg, tokenizer = create_om_cfg(FT_API_args)

input_folder = FT_API_args.train_data_path
output_folder = FT_API_args.save_folder
concat_tokens = FT_API_args.context_length
tokenizer_name = FT_API_args.model

# Run convert_text_to_mds.py and dump MDS dataset to "save_folder"
args = parse_args(tokenizer, concat_tokens, output_folder, input_folder)
convert_text_to_mds(tokenizer_name=args.tokenizer,
                    output_folder=temporary_mds_output_path,
                    input_folder=args.input_folder,
                    concat_tokens=args.concat_tokens,
                    eos_text=args.eos_text,
                    bos_text=args.bos_text,
                    no_wrap=args.no_wrap,
                    compression=args.compression,
                    processes=args.processes,
                    reprocess=args.reprocess,
                    args_str=_args_str(args))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Alternative: Delta Ingestion 
# MAGIC Once you have credentials set up with dbutils.secret or init script, You can ingest the folder of txt files and have the schema automatically inferred. The result is a spark dataframe and can be converted to MDS while Streaming's utility

# COMMAND ----------

dbutils.fs.ls(FT_API_args.train_data_path)

output_location = FT_API_args.train_data_path + '/*.txt'
df = spark.sql("SELECT * FROM read_files('%s')" % output_location).withColumnRenamed('value', 'text')
df.show()

mds_kwargs = {
    'out': temporary_mds_output_path,
    'columns': {
        'tokens': 'bytes'
    },
    'keep_local': True
}
udf_kwargs = {
    'concat_tokens': FT_API_args.context_length,
    'tokenizer': FT_API_args.model, 
    'eos_text': '',
    'compression': 'zstd',
    'no_wrap': False,
    'bos_text': '',
}

dataframe_to_mds(df,
                  merge_index=True,
                  mds_kwargs=mds_kwargs,
                  udf_iterable=pandas_processing_fn,
                  udf_kwargs=udf_kwargs)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validation

# COMMAND ----------

print("Num examples:", len(df))
print("First example:")
for ex in df['text']: 
    print(ex)
    print() 
    break 

if integrity_check(temporary_mds_output_path): 
    raise ValueError("MDS has not been created correctly. There are missing shards")

# Sanity Check
import numpy as np
from streaming import StreamingDataset
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace
dataset = StreamingDataset(local=mds_output_path, shuffle=False)
for i in range(5):
    l = np.frombuffer(dataset[i]['tokens'], dtype=np.int64)
    print(''.join(tokenizer.decode(l)))
    print()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Cost Estimation

# COMMAND ----------

MAX_TOKENS_PER_EXAMPLE = FT_API_args.context_length if FT_API_args.context_length is not None else 4096
TARGET_EPOCHS = FT_API_args.training_duration if FT_API_args.training_duration is not None else 1 
n_epochs = TARGET_EPOCHS
n_train_examples = len(raw_dataset)

batch_tokens = token_counts_and_validation(FT_API_args)
n_billing_tokens_in_dataset = sum(batch_tokens)

print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

# COMMAND ----------



# COMMAND ----------


