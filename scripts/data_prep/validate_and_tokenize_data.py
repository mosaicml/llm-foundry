# Databricks notebook source
# MAGIC %md
# MAGIC # FM FT API: Validation and Cost Estimation
# MAGIC
# MAGIC #### Usage Scenario:
# MAGIC This notebook goes hand-in-hand with Databricks-Mosaicml's FT API. Our customers may find it useful in scenarios where there is a risk of data being malformed. It acts as a preventive measure to ensure data integrity and helps in cost assessment for the fine-tuning process.
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
# MAGIC - For the reference, FT API expects following
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
# MAGIC     disable_credentials_check: Optional[bool] = None,
# MAGIC     timeout: Optional[float] = 10,
# MAGIC     future: Literal[False] = False,
# MAGIC }
# MAGIC ``` 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install llmfoundry Validation Branch

# COMMAND ----------

# %pip install git+https://github.com/mosaicml/llm-foundry.git@byod/data_validation
%pip install git+https://github.com/XiaohanZhangCMU/llm-foundryX.git@validation 

# COMMAND ----------

dbutils.library.restartPython()

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
from llmfoundry.utils import (create_om_cfg, token_counts_and_validation, token_counts, 
        check_HF_datasets, is_hf_dataset_path, is_uc_delta_table,
        pandas_processing_fn, integrity_check, convert_text_to_mds,
        _args_str, plot_hist)
from llmfoundry.data import ConcatTokensDataset

from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader
from streaming.base.converters import dataframe_to_mds

# COMMAND ----------

# MAGIC %md 
# MAGIC # Instruction Fine Tuning

# COMMAND ----------

# MAGIC %md 
# MAGIC #### User Defines

# COMMAND ----------

FT_API_args = Namespace(
    model='EleutherAI/gpt-neox-20b',
    train_data_path= 'main.streaming.random_large_table', # '/Volumes/main/mosaic_hackathon/managed-volume/IFT/train.jsonl', # 'tatsu-lab/alpaca/train', # , # 'tatsu-lab/alpaca/train',  # 'mosaicml/dolly_hhrlhf/train', # tatsu-lab/alpaca/train',
    task_type='INSTRUCTION_FINETUNE',
    training_duration=3,
    context_length=2048,
)

temporary_jsonl_data_path = '/tmp/ft_data_11Jan24_2/train'
# os.environ['HF_ASSETS_CACHE'] = '/tmp/'
# os.environ['HF_HOME'] = '/tmp/'
# os.environ['HF_HUB_CACHE'] = '/tmp/'
os.environ['HF_DATASETS_CACHE'] = '/tmp/'
os.makedirs(temporary_jsonl_data_path, exist_ok=True)

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

if is_hf_dataset_path(FT_API_args.train_data_path):
    check_HF_datasets(FT_API_args.train_data_path)
    dataset_id, split = '/'.join(FT_API_args.train_data_path.split('/')[:2]), FT_API_args.train_data_path.split('/')[-1]    
    raw_dataset = datasets.load_dataset(dataset_id, split=split)       
else:
    if is_uc_delta_table(FT_API_args.train_data_path):    
        df = spark.read.table(FT_API_args.train_data_path).toPandas()
        df.to_json(os.path.join(temporary_jsonl_data_path, 'data.jsonl'), orient='records', lines=True)
        raw_dataset = datasets.Dataset.from_pandas(df) 
        FT_API_args.train_data_path = temporary_jsonl_data_path
    else: 
        # train_data_path is a jonsl file (local/remote)
        from composer.utils import dist, get_file, parse_uri 
        data_path = FT_API_args.train_data_path 
        backend, _, _ = parse_uri(data_path)
        if backend not in ['', None]: # It's a remote path, download before loading it
            with tempfile.TemporaryDirectory() as tmp_dir:
                destination = os.path.join(tmp_dir, 'data.jsonl')
                get_file(data_path, destination)
                df = pd.read_json(destination, orient='records', lines=True)    
        else: 
            df = pd.read_json(data_path, orient='records', lines=True)    

        raw_dataset = datasets.Dataset.from_pandas(df)
        FT_API_args.train_data_path = os.path.dirname(data_path)

if raw_dataset is None: 
    raise RuntimeError("Can't find a proper ingestion method")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validation

# COMMAND ----------

# Initial dataset stats
print("Num examples:", len(raw_dataset))
print("First example:")
for ex in raw_dataset: 
    print(ex)
    print() 
    break 

_ALLOWED_RESPONSE_KEYS = {'response', 'completion'}
_ALLOWED_PROMPT_KEYS = {'prompt'}
format_errors = defaultdict(int)

for ex in raw_dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1 
        continue 
    
    found = False 
    for key in _ALLOWED_PROMPT_KEYS:
        prompts = ex.get(key, None)
        if prompts:
            found = True 
    if not found: 
        format_errors["missing_prompt"] += 1

    found = False
    for key in _ALLOWED_RESPONSE_KEYS:        
        responses = ex.get("response", None)
        if responses: 
            found = True 
    if not found:
        format_errors["missing_response"] += 1
        
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

n_epochs = FT_API_args.training_duration if FT_API_args.training_duration is not None else 1 
batch_tokens = token_counts3(FT_API_args)
n_billing_tokens_in_dataset = sum(batch_tokens['ntokens'])

print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

# COMMAND ----------

plot_hist(pd.Series(batch_tokens['ntokens']))

# COMMAND ----------

# all_tokens = token_counts_and_validation(FT_API_args)
# plot_hist(pd.Series(all_tokens))
# pd.Series(all_tokens).max(), max(batch_tokens['ntokens'])

# COMMAND ----------

# MAGIC %md 
# MAGIC # Continued Pretrain

# COMMAND ----------

# MAGIC %md 
# MAGIC #### User Defines

# COMMAND ----------

FT_API_args = Namespace(
    model='EleutherAI/gpt-neox-20b',
    train_data_path= '/Volumes/main/mosaic_hackathon/managed-volume/ABT',
    task_type='CONTINUED_PRETRAIN',
    training_duration=3,
    context_length=2048,
)
temporary_mds_output_path = '/tmp/xiaohan-test-11Jan24_2/test_mds'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ingestion, Tokenization and Materialization
# MAGIC
# MAGIC CPT takes a folder of txt files as input. It tokenize the text fields and materialize as a streaming dataset of MDS format. 
# MAGIC
# MAGIC FT API uses [llmfoundry/scripts/data_prep/convert_text_to_mds.py](https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/convert_text_to_mds.py) to download all the txt files and convert them to MDS. 
# MAGIC
# MAGIC In this notebook, we provide two additional approaches via Spark and Dask. 
# MAGIC
# MAGIC **Warning** CPT datasets are normally much larger than IFT, so the tokenization and materialization can be very time consuming. 

# COMMAND ----------

# MAGIC %md 
# MAGIC **1. Delta Ingestion --> Spark Dataframe:** 
# MAGIC
# MAGIC If you don't have a single-user-assigned cluster and DBR < 14.3, move on to option-2. Otherwise, you can leverage Delta Ingestion's tools to ingest the folder of txt files as a Spark dataframe and have the schema automatically inferred. 

# COMMAND ----------

dbutils.fs.ls(FT_API_args.train_data_path)

output_location = FT_API_args.train_data_path + '/*.txt'
df = spark.sql("SELECT * FROM read_files('%s')" % output_location).withColumnRenamed('value', 'text')
df.show(2)
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
# MAGIC **2. Dask.bag --> Dask.DataFrame:**  
# MAGIC
# MAGIC If you are on UC enabled clusters where mapInPandas does not work, you can try Dask. Dask uses the current node as a ```Local Cluster```

# COMMAND ----------

import dask.bag as db

input_folder = FT_API_args.train_data_path
pattern = input_folder + '/*.txt'
b = db.read_text(pattern, linedelimiter='\n', blocksize='128MiB')
df = b.to_dataframe(columns = ['text'])
df = df[df.text != '\n']

mds_kwargs = {
    'out': temporary_mds_output_path,
    'columns': {
        'tokens': 'bytes'
    },
    'keep_local': True, 
}
udf_kwargs = {
    'concat_tokens': FT_API_args.context_length,
    'tokenizer': FT_API_args.model, 
    'eos_text': '',
    'compression': 'zstd',
    'no_wrap': False,
    'bos_text': '',
}
df_to_mds(df,
          merge_index=True,
          mds_kwargs=mds_kwargs,
          udf_iterable=pandas_processing_fn2,
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
mds_dataset = StreamingDataset(local=mds_output_path, shuffle=False)
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

n_billing_tokens_in_dataset = len(mds_dataset) * FT_API_args.context_length 
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
