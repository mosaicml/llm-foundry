from argparse import ArgumentParser, Namespace
from typing import Iterable, cast, List, Tuple
import os

from streaming import MDSWriter
from tqdm import tqdm
from composer.utils import (ObjectStore, S3ObjectStore, maybe_create_object_store_from_uri,
                        parse_uri)
from llmfoundry.data import ConcatTokensDataset
from transformers import AutoTokenizer

import math
from multiprocessing import Pool
from glob import glob
import json
import tempfile
import hashlib

from llmfoundry.data.packing import build_dataloader

from scripts.data_prep.utils import generate_samples

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=64,
        help='The maximum number of workers to use for MDS writing')
    parser.add_argument('--output_folder',
                        type=str,
                        required=True,
                        help='The folder to write output to')
    parser.add_argument('--train_index_path',
                        type=str,
                        required=True,
                        help='The file that lists all the data paths to read from for training')
    parser.add_argument('--eval_index_path',
                        type=str,
                        required=True,
                        help='The file that lists all the data paths to read from for evaluation')
    parser.add_argument('--compression',
                        type=str,
                        default='zstd',
                        help='The compression algorithm to use for MDS writing')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer',
                        type=str,
                        help='The name of the tokenizer to use')
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples')
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples')
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples')
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=1,
        help='The number of processes to use to download and convert the dataset'
    )

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

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
        max_workers: int,
        compression: str,
        ) -> Iterable:
    objs_per_group = math.ceil(len(object_names) / n_groups)
    for group, i in enumerate(range(0, len(object_names), objs_per_group)):
        output_folder = os.path.join(output_root, str(group))
        yield (
            object_names[i:i + objs_per_group], output_folder, input_folder, tokenizer_name, concat_tokens, eos_text, bos_text, no_wrap, max_workers, compression)

def download_and_convert(
        file_names: List[str], 
        output_folder: str, 
        input_folder: str,    
        tokenizer_name: str, 
        concat_tokens: int,
        eos_text: str,
        bos_text: str,
        no_wrap: bool,
        max_workers: int,
        compression: str,
    ):
    object_store = maybe_create_object_store_from_uri(input_folder)
    _, _, folder_prefix = parse_uri(input_folder)

    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:  
        downloading_iter = DownloadingIterable(file_names, folder_prefix, tmp_dir, object_store)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = concat_tokens # Hack to remove hugging face warning 

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=downloading_iter,
            max_length=concat_tokens,
            tokenizer=tokenizer,
            eos_text=eos_text,
            bos_text=bos_text,
            no_wrap=no_wrap,
        )

        loader = build_dataloader(dataset=dataset, batch_size=512)
        samples = generate_samples(loader)
        columns = {'tokens': 'bytes'}

        print(f'Converting to MDS format...')
        with MDSWriter(out=output_folder,
                        columns=columns,
                        compression=False,
                        max_workers=max_workers,
                        compression=compression
                        ) as out:
            for sample in tqdm(samples):
                out.write(sample)

def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id. 
    From https://github.com/mosaicml/streaming/blob/main/examples/multiprocess_dataset_conversion.ipynb

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
    From https://github.com/mosaicml/streaming/blob/main/examples/multiprocess_dataset_conversion.ipynb

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

def is_remote_path(path: str) -> bool:
    backend, bucket, _ = parse_uri(path)
    return backend != '' or bucket != ''

def is_already_processed(output_root: str, done_file_name: str, args: Namespace, objects_hash: str, is_remote_output: bool) -> bool:
    # Retrieve the done file contents
    if is_remote_output:
        # Download and read the done file from the remote object store
        output_object_store = maybe_create_object_store_from_uri(output_root)
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, done_file_name)
                output_object_store.download_object(os.path.join(output_folder_prefix, done_file_name), done_file)
                done_file_contents = open(done_file).readlines()
        except FileNotFoundError:
            return False
    else:
        # Read the local done file
        done_file = os.path.join(output_root, done_file_name)
        if not os.path.isfile(done_file):
            return False
        done_file_contents = open(done_file).readlines()

    # Compare the hash and the arguments
    prev_objects_hash = done_file_contents[0]
    return objects_hash == prev_objects_hash # TODO: Compare arg namespace

# Initialize the worker process
def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')


# def main(args: Namespace):
def main(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    max_workers: int,
    compression: str,
    processes: int,
):
    done_file_name = 'done'

    is_remote_output = is_remote_path(output_folder)

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory().name if is_remote_output and processes > 1 else output_folder

    input_object_store = maybe_create_object_store_from_uri(input_folder)
    _, _, folder_prefix = parse_uri(input_folder)
    
    object_names, objects_hash = get_object_names_and_hash(input_object_store, folder_prefix)

    # Check if the text files in the bucket have already been processed.
    if is_already_processed(output_folder, done_file_name, Namespace(), objects_hash, is_remote_output):
        print(f'Input folder {input_folder} is already processed at {output_folder}.')
        # TODO: Add flag to automatically reprocess
        return

    if processes > 1:
        # Download and convert the text files in parallel
        args = get_task_args(object_names, local_output_folder, input_folder, processes,
                             tokenizer_name, concat_tokens, eos_text, bos_text, no_wrap, compression, max_workers)
        with Pool(initializer=init_worker, processes=processes) as pool:
            pool.starmap(download_and_convert, args)
        
        # Merge the mds shards from each of the processes into a single folder
        merge_shard_groups(local_output_folder)
    else:
        download_and_convert(object_names, local_output_folder, input_folder, 
                             tokenizer_name, concat_tokens, eos_text, bos_text, no_wrap, compression, max_workers)

    # Write a done file with the args and the hash
    with open(os.path.join(local_output_folder, done_file_name), 'w') as done_file:
        done_file.write(str(objects_hash))
        # done_file.write('\n')
        # done_file.write(str(args))
        # TODO: write arguments to file
        # TODO: write 

    if is_remote_output and processes > 1:
        # Upload the local output to the remote location
        output_object_store = maybe_create_object_store_from_uri(output_folder)
        _, _, output_folder_prefix = parse_uri(output_folder)
        pattern = os.path.join(local_output_folder, '*')
        files_to_upload = sorted(glob(pattern))

        # TODO: Use multi threading to upload files?
        for local_path in files_to_upload:
            assert not os.path.isdir(local_path)
            remote_path = os.path.join(output_folder_prefix, os.path.basename(local_path))
            output_object_store.upload_object(remote_path, local_path)

if __name__ == '__main__':
    args = parse_args()
    main(
        tokenizer_name=args.tokenizer,
        output_folder=args.out_root,
        input_folder=args.in_root,
        dataset_subset=args.dataset_subset,
        concat_tokens=args.concat_tokens,
        eos_text=args.eos_text,
        bos_text=args.bos_text,
        no_wrap=args.no_wrap,
        max_workers=args.max_workers,
        compression=args.compression,
        processes=args.processes
    )
    