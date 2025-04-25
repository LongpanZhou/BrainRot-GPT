import os
import tiktoken
import numpy as np

from tqdm import tqdm
from datasets import load_dataset

# Code stolen from: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py#L45

# Global constants
NUM_PROC = os.cpu_count()>>1
SAMPLE_NAME = "sample-10BT"
BATCH_SIZE = 1024

print("Downloading dataset...")
# Download dataset - Does not show progress bar...
fw = load_dataset("HuggingFaceFW/fineweb", name=SAMPLE_NAME, split="train", num_proc=NUM_PROC)
#Set streaming = True if you do not want to download locally
#Set cache_dir = LOCAL_DATASET_DIR if you want to download to local folder
fw_split = fw.train_test_split(test_size=0.2, shuffle=True)

# Encoder
enc = tiktoken.get_encoding("cl100k_base")
eot = enc.eot_token

# Process Func
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = fw_split.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing",
    num_proc=NUM_PROC,
)

# Save the tokenized dataset
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=np.uint32, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()