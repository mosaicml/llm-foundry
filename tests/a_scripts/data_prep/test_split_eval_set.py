import os
import json
import pytest
import hashlib
from unittest.mock import patch

from llmfoundry.command_utils import split_eval_set_from_args, split_examples

# Default values
OUTPUT_DIR = "tmp-split"
TMPT_DIR = "tmp-t"
DATA_PATH_SPLIT = "train"
EVAL_SPLIT_RATIO = 0.1
DEFAULT_FILE = TMPT_DIR + "/train-00000-of-00001.jsonl"


def calculate_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash


def count_lines(filepath: str) -> int:
    with open(filepath, "r") as f:
        return sum(1 for _ in f)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_module():
    # Setup: create local testing file
    os.makedirs(TMPT_DIR, exist_ok=True)
    with open(DEFAULT_FILE, "w") as f:
        for i in range(1000):
            f.write(json.dumps({"prompt": "hello world " + str(i), "response": "hi you!"}) + "\n")
    yield

    # Teardown: clean up output and tmp directories
    os.system(f"rm -rf {OUTPUT_DIR}")
    os.system(f"rm -rf {TMPT_DIR}")


def test_basic_split():
    """Test basic functionality on local file"""
    output_path = os.path.join(OUTPUT_DIR, "basic-test")
    split_eval_set_from_args(TMPT_DIR, DATA_PATH_SPLIT, output_path, EVAL_SPLIT_RATIO)
    assert os.path.isfile(os.path.join(output_path, "train.jsonl"))
    assert os.path.isfile(os.path.join(output_path, "eval.jsonl"))


def test_basic_split_output_exists():
    """Test that split overwrites existing files in output directory"""
    output_path = os.path.join(OUTPUT_DIR, "basic-test")
    os.makedirs(output_path, exist_ok=True)
    train_file = os.path.join(output_path, "train.jsonl")
    eval_file = os.path.join(output_path, "eval.jsonl")
    with open(train_file, "w") as f:
        f.write("existing file train")
    with open(eval_file, "w") as f:
        f.write("existing file eval")
    old_train_hash = calculate_file_hash(train_file)
    old_eval_hash = calculate_file_hash(eval_file)
    split_eval_set_from_args(
        TMPT_DIR,
        DATA_PATH_SPLIT,
        output_path,
        EVAL_SPLIT_RATIO,
    )
    assert calculate_file_hash(train_file) != old_train_hash
    assert calculate_file_hash(eval_file) != old_eval_hash


def test_max_eval_samples():
    """Test case where max_eval_samples < eval_split_ratio * total samples"""
    output_path = os.path.join(OUTPUT_DIR, "max-eval-test")
    max_eval_samples = 50
    split_eval_set_from_args(
        TMPT_DIR,
        DATA_PATH_SPLIT,
        output_path,
        EVAL_SPLIT_RATIO,
        max_eval_samples,
    )
    eval_lines = count_lines(os.path.join(output_path, "eval.jsonl"))
    assert eval_lines == max_eval_samples


def test_eval_split_ratio():
    """Test case where max_eval_samples is not used"""
    output_path = os.path.join(OUTPUT_DIR, "eval-split-test")
    split_eval_set_from_args(TMPT_DIR, DATA_PATH_SPLIT, output_path, EVAL_SPLIT_RATIO)
    original_data_lines = count_lines(DEFAULT_FILE)
    eval_lines = count_lines(os.path.join(output_path, "eval.jsonl"))
    assert abs(eval_lines - EVAL_SPLIT_RATIO * original_data_lines) < 1  # allow for rounding errors


def test_seed_consistency():
    """Test if the same seed generates consistent splits"""
    output_path_1 = os.path.join(OUTPUT_DIR, "seed-test-1")
    output_path_2 = os.path.join(OUTPUT_DIR, "seed-test-2")
    split_examples(DEFAULT_FILE, output_path_1, EVAL_SPLIT_RATIO, seed=12345)
    split_examples(DEFAULT_FILE, output_path_2, EVAL_SPLIT_RATIO, seed=12345)
    train_hash_1 = calculate_file_hash(os.path.join(output_path_1, "train.jsonl"))
    train_hash_2 = calculate_file_hash(os.path.join(output_path_2, "train.jsonl"))
    eval_hash_1 = calculate_file_hash(os.path.join(output_path_1, "eval.jsonl"))
    eval_hash_2 = calculate_file_hash(os.path.join(output_path_2, "eval.jsonl"))

    assert train_hash_1 == train_hash_2
    assert eval_hash_1 == eval_hash_2

    output_path_3 = os.path.join(OUTPUT_DIR, "seed-test-3")
    split_examples(DEFAULT_FILE, output_path_3, EVAL_SPLIT_RATIO, seed=54321)
    train_hash_3 = calculate_file_hash(os.path.join(output_path_3, "train.jsonl"))
    eval_hash_3 = calculate_file_hash(os.path.join(output_path_3, "eval.jsonl"))

    assert train_hash_1 != train_hash_3
    assert eval_hash_1 != eval_hash_3


def test_hf_data_split():
    """Test splitting a dataset from Hugging Face"""
    output_path = os.path.join(OUTPUT_DIR, "hf-split-test")
    split_eval_set_from_args(
        "databricks/databricks-dolly-15k", "train", output_path, EVAL_SPLIT_RATIO
    )
    assert os.path.isfile(os.path.join(output_path, "train.jsonl"))
    assert os.path.isfile(os.path.join(output_path, "eval.jsonl"))
    assert count_lines(os.path.join(output_path, "train.jsonl")) > 0
    assert count_lines(os.path.join(output_path, "eval.jsonl")) > 0


def _mock_get_file(remote_path: str, data_path: str, overwrite: bool):
    with open(data_path, "w") as f:
        for i in range(1000):
            f.write(json.dumps({"prompt": "hello world " + str(i), "response": "hi you!"}) + "\n")


def test_remote_store_data_split():
    """Test splitting a dataset from a remote store"""
    output_path = os.path.join(OUTPUT_DIR, "remote-split-test")
    with patch("composer.utils.get_file", side_effect=_mock_get_file) as mock_get_file:
        split_eval_set_from_args(
            "dbfs:/Volumes/test/test/test.jsonl",
            "unique-split-name",
            output_path,
            EVAL_SPLIT_RATIO,
        )
        mock_get_file.assert_called()

    assert os.path.isfile(os.path.join(output_path, "train.jsonl"))
    assert os.path.isfile(os.path.join(output_path, "eval.jsonl"))
    assert count_lines(os.path.join(output_path, "train.jsonl")) > 0
    assert count_lines(os.path.join(output_path, "eval.jsonl")) > 0


def test_missing_delta_file_error():
    # expects file 'TMPT_DIR/missing-00000-of-00001.jsonl
    with pytest.raises(FileNotFoundError):
        split_eval_set_from_args(TMPT_DIR, "missing", OUTPUT_DIR, EVAL_SPLIT_RATIO)


def test_unknown_file_format_error():
    with pytest.raises(ValueError):
        split_eval_set_from_args("s3:/path/to/file.jsonl", "train", OUTPUT_DIR, EVAL_SPLIT_RATIO)
