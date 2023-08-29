from argparse import Namespace
from typing import Dict, Iterable, Optional, cast, List, Tuple
import os

from streaming import MDSWriter
from tqdm import tqdm
from composer.utils import (ObjectStore, S3ObjectStore, maybe_create_object_store_from_uri,
                        parse_uri)
from llmfoundry.data import ConcatTokensDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import math
from multiprocessing import Pool
from glob import glob
import json
import tempfile
import hashlib

# check last args and input folder modification
# if last args are the same as current args and the last modified date is the same,

def build_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size
    )

def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


class DownloadingIterable:
    def __init__(
        self,
        object_names: List[str],
        input_folder_prefix: str,
        output_folder: str,
        object_store: ObjectStore,
    ):
        """Iterable that downloads files from an object store before yielding.

        text samples.

        Args:
            identifiers (List[str]): List of identifiers (<doc id>|||<ticker>|||<report_date>) to iterate over
            input_folder_prefix (str): Object store prefix to download from
            output_folder (str): Local folder to write downloaded files to
            object_store (ObjectStore): Object store to download from
        """
        self.object_names = object_names
        self.object_store = object_store
        self.input_folder_prefix = input_folder_prefix
        self.output_folder = output_folder

    def __iter__(self):
        for object_name in self.object_names:
            output_filename = os.path.join(self.output_folder, os.path.relpath(object_name, start=self.input_folder_prefix))
            self.object_store.download_object(
                object_name=object_name,
                filename = output_filename,
                overwrite=True
            )

            with open(output_filename) as _txt_file:
                txt = _txt_file.read()
            yield {'text': txt}


def get_object_names_and_hash(object_store: ObjectStore, folder_prefix: str) -> Tuple[List[str], str]:
    # TODO: Support other object store backends
    object_store = cast(S3ObjectStore , object_store)
    paginator = object_store.client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=object_store.bucket, Prefix=folder_prefix)

    object_names = []
    objects_hash = hashlib.sha1()
    for page in pages:
        for obj in page['Contents']:
            name, etag = obj['Key'], obj['ETag']
            if name.endswith('.txt'):
                object_names.append(name)
                objects_hash.update(str.encode(etag)) # Update the hash with the byte-encoded etag
    print(f'Found {len(object_names)} text files')
    return object_names, objects_hash.hexdigest()

def get_task_args(object_names: List[str], output_root: str, input_folder: str, n_groups: int) -> Iterable[tuple[List[str], str]]:
    objs_per_group = math.ceil(len(object_names) / n_groups)
    for group, i in enumerate(range(0, len(object_names), objs_per_group)):
        output_folder = os.path.join(output_root, str(group))
        yield (object_names[i:i + objs_per_group], output_folder, input_folder)

def download_and_convert(inputs: tuple[List[str], str]):
    file_names, output_folder, input_folder = inputs
    object_store = maybe_create_object_store_from_uri(input_folder)
    _, _, folder_prefix = parse_uri(input_folder)
    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:  
        downloading_iter = DownloadingIterable(file_names, folder_prefix, tmp_dir, object_store)
        tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b', max_length=2048)

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=downloading_iter,
            max_length=2048,
            tokenizer=tokenizer,
            eos_text='<|endoftext|>',
            bos_text='',
            no_wrap=None,
        )

        loader = build_dataloader(dataset=dataset, batch_size=512)
        samples = generate_samples(loader)
        columns = {'tokens': 'bytes'}

        print(f'Converting to MDS format...')
        with MDSWriter(out=output_folder,
                        columns=columns,
                        compression=False,
                        ) as out:
            for sample in tqdm(samples):
                # print(tokenizer.decode(np.frombuffer(sample['tokens'], dtype=int)))
                out.write(sample)

def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id.

    Args:
        basename (str): Old basename of file.
        shard_id (int): New shard ID.

    Returns:
        str: New basename of file.
    """
    parts = basename.split('.')
    parts[1] = f'{shard_id:05}'
    return '.'.join(parts)


def merge_shard_groups(root: str) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        root (str): Root directory.
    """
    pattern = os.path.join(root, '*')
    subdirs = sorted(glob(pattern))
    shard_id = 0
    infos = []
    for subdir in subdirs:
        index_filename = os.path.join(subdir, 'index.json')
        obj = json.load(open(index_filename))
        for info in obj['shards']:
            old_basename = info['raw_data']['basename']
            new_basename = with_id(old_basename, shard_id)
            info['raw_data']['basename'] = new_basename

            if info['zip_data'] is not None:
                old_basename = info['zip_data']['basename']
                new_basename = with_id(old_basename, shard_id)
                info['zip_data']['basename'] = new_basename

            old_filename = os.path.join(subdir, old_basename)
            new_filename = os.path.join(root, new_basename)
            assert not os.rename(old_filename, new_filename)

            shard_id += 1
            infos.append(info)

        assert not os.remove(index_filename)
        assert not os.rmdir(subdir)

    index_filename = os.path.join(root, 'index.json')
    obj = {
        'version': 2,
        'shards': infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, 'w') as out:
        out.write(text)

def is_local_path(path: str) -> bool:
    backend, bucket, _ = parse_uri(path)
    return backend == '' and bucket == ''

def is_already_processed(output_root: str, done_file_name: str, args: Namespace, objects_hash: str, is_remote_output: bool) -> bool:
    done_file_contents = None 
    if is_remote_output:
        output_object_store = maybe_create_object_store_from_uri(output_root)
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, done_file_name)
                obj = output_object_store.download_object(os.path.join(output_folder_prefix, done_file_name), done_file)
                done_file_contents = open(done_file).readlines()
        except FileNotFoundError:
            return False
    else:
        done_file = os.path.join(output_root, done_file_name)
        if not os.path.isfile(done_file):
            return False
        done_file_contents = open(done_file).readlines()
    try:
        prev_objects_hash = done_file_contents[0]
        if objects_hash == prev_objects_hash:
            return True # TODO: Compare arg namespace
    except ValueError:
        pass
    return False

# Initialize the worker process
def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')


# def main(args: Namespace):
def main():
    n_processes = 8
    n_groups = 8
    output_root = 's3://mosaicml-internal-checkpoints-shared/irene/test-output'
    input_folder = 's3://mosaicml-internal-checkpoints-shared/irene/sec-filings-large-train'
    done_file_name = 'done'

    is_remote_output = not is_local_path(output_root)
    local_output_root = tempfile.TemporaryDirectory().name if is_remote_output else output_root

    input_object_store = maybe_create_object_store_from_uri(input_folder)
    _, _, folder_prefix = parse_uri(input_folder)
    
    object_names, objects_hash = get_object_names_and_hash(input_object_store, folder_prefix)

    if is_already_processed(output_root, done_file_name, Namespace(), objects_hash, is_remote_output):
        print(f'Input folder {input_folder} is already processed at {output_root}.')
        # TODO: Add flag to automatically reprocess
        return

    tasks = get_task_args(object_names, local_output_root, input_folder, n_groups)

    with Pool(initializer=init_worker, processes=n_processes) as pool:
        pool.map(download_and_convert, tasks)

    merge_shard_groups(local_output_root)

    # Write a done file with the args and the hash
    with open(os.path.join(local_output_root, done_file_name), 'w') as done_file:
        done_file.write(str(objects_hash))
        # done_file.write('\n')
        # done_file.write(str(args))
        # TODO: write arguments to file
        # TODO: write 

    if is_remote_output:
        # Upload the local output to the remote location
        output_object_store = maybe_create_object_store_from_uri(output_root)
        _, _, output_folder_prefix = parse_uri(output_root)
        pattern = os.path.join(local_output_root, '*')
        files_to_upload = sorted(glob(pattern))
        for local_path in files_to_upload:
            assert not os.path.isdir(local_path)
            remote_path = os.path.join(output_folder_prefix, os.path.basename(local_path))
            output_object_store.upload_object(remote_path, local_path)


if __name__ == '__main__':
    main()