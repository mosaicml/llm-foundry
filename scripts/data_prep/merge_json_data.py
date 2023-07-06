import os
import json
import shutil
import argparse

def count_mds_files(base_dir):
    return sum(1 for filename in os.listdir(base_dir) if filename.endswith(".mds"))

def increment_filenames_and_update_index(base_dir, increment):
    index_path = os.path.join(base_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    new_filenames = {}

    for filename in sorted(os.listdir(base_dir), reverse=True):
        if filename.startswith("shard."):
            basename, extension = os.path.splitext(filename)
            prefix, number = basename.split(".")
            old_number = int(number)
            new_number = old_number + increment
            old_filename = f"{prefix}.{str(old_number).zfill(5)}{extension}"
            new_filename = f"{prefix}.{str(new_number).zfill(5)}{extension}"
            os.rename(os.path.join(base_dir, filename), os.path.join(base_dir, new_filename))
            new_filenames[old_filename] = new_filename

    for old_filename, new_filename in new_filenames.items():
        for shard in index["shards"]:
            if shard["raw_data"]["basename"] == old_filename:
                shard["raw_data"]["basename"] = new_filename


    with open(index_path, "w") as f:
        print("index_path",index_path)
        json.dump(index, f)

def merge_indexes(dir1, dir2_updated):
    with open(os.path.join(dir1, "index.json"), "r") as f1, open(os.path.join(dir2_updated, "index.json"), "r") as f2:
        index1 = json.load(f1)
        index2 = json.load(f2)
    index1["shards"].extend(index2["shards"])
    with open(os.path.join(dir2_updated, "index.json"), "w") as f:
        json.dump(index1, f)

def main(dir1, dir2, dest):
    # Copy contents of dir2 to dest
    shutil.copytree(dir2, dest)

    increment_value = count_mds_files(dir1)
    print("increment_value",increment_value)

    # Update file name and update index
    increment_filenames_and_update_index(dest, increment_value)

    # First, merge the contents of index.json
    merge_indexes(dir1, dest)

    # Copy contents of dir1 to dest except index.json
    for item in os.listdir(dir1):
        if item != "index.json":
            s = os.path.join(dir1, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", help="First directory to process")
    parser.add_argument("dir2", help="Second directory to process")
    parser.add_argument("dest", help="Destination directory")
    args = parser.parse_args()
    main(args.dir1, args.dir2, args.dest)