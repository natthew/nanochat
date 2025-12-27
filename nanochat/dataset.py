"""
The base/pretraining dataset is a set of jsonl files (optionally gzipped).
This file contains utilities for:
- iterating over the jsonl files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import json
import gzip
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/common-pile/pre_1929_books/resolve/main/data/documents"
MAX_SHARD = 99 # adjust this based on the actual number of shards available
index_to_filename = lambda index: f"{index:05d}_public_library_1929.jsonl.gz" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all jsonl files.
    Name kept as list_parquet_files for backward compatibility. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    jsonl_files = sorted([
        f for f in os.listdir(data_dir)
        if (f.endswith('.jsonl') or f.endswith('.jsonl.gz')) and not f.endswith('.tmp')
    ])
    jsonl_paths = [os.path.join(data_dir, f) for f in jsonl_files]
    return jsonl_paths

def parquets_iter_batched(split, start=0, step=1, batch_size=1024):
    """
    Iterate through the dataset, in batches for efficiency.
    - split can be "train" or "val". the last jsonl file will be val.
    - start/step are useful for skipping batches in DDP. e.g. start=rank, step=world_size
    - batch_size: number of documents per batch (default 1024, similar to parquet row_groups)
    Name kept as parquets_iter_batched for backward compatibility.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    jsonl_paths = list_parquet_files()
    jsonl_paths = jsonl_paths[:-1] if split == "train" else jsonl_paths[-1:]

    for filepath in jsonl_paths:
        # Determine if file is gzipped
        is_gzipped = filepath.endswith('.gz')
        open_fn = gzip.open if is_gzipped else open
        mode = 'rt' if is_gzipped else 'r'

        # Read all lines from the file and batch them
        with open_fn(filepath, mode, encoding='utf-8') as f:
            batch = []
            batch_idx = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    batch.append(doc['text'])

                    # When we've accumulated batch_size documents, yield if it's our turn (DDP)
                    if len(batch) >= batch_size:
                        if batch_idx >= start and (batch_idx - start) % step == 0:
                            yield batch
                        batch = []
                        batch_idx += 1
                except (json.JSONDecodeError, KeyError) as e:
                    # Skip malformed lines
                    continue

            # Yield any remaining documents in the last batch
            if batch and batch_idx >= start and (batch_idx - start) % step == 0:
                yield batch

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-1929 Books dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
