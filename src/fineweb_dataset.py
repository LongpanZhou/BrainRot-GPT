import os
import tiktoken
import numpy as np

from tqdm import tqdm
from datasets import load_dataset, load_from_disk

# Code stolen from: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py#L45

# Global constants
NUM_PROC = os.cpu_count()                                                               # Number of processes to use for tokenization
SAMPLE_NAME = "sample-10BT"                                                             # Sample name from HuggingFaceFW/Fineweb dataset
ENCODER_NAME = "cl100k_base"                                                            # Encoder name
PATH_NAME = os.path.join(os.path.dirname(__file__), f'{SAMPLE_NAME}_{ENCODER_NAME}')    # Path to save the dataset
BATCH_SIZE = 1024                                                                       # Batch size for tokenization
ALL_IN_ONE = False                                                                      # Save all data in one file. Set to True if you have enough memory

# Download dataset - Does not show progress bar...
print("Downloading dataset...")
fw = load_dataset("HuggingFaceFW/fineweb", name=SAMPLE_NAME, split="train", num_proc=NUM_PROC)
#Set streaming = True if you do not want to download locally
#Set cache_dir = LOCAL_DATASET_DIR if you want to download to local folder
fw_split = fw.train_test_split(test_size=0.2, shuffle=True)

# Encoder
enc = tiktoken.get_encoding(ENCODER_NAME)
eot = enc.eot_token

# Process Func
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

# Load or tokenize dataset
if not os.path.exists(PATH_NAME):
    print("Tokenizing dataset...")
    tokenized = fw_split.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )
    tokenized.save_to_disk(PATH_NAME)
else:
    print("Loading tokenized dataset from disk...")
    tokenized = load_from_disk(PATH_NAME)

# Save the tokenized dataset
for split, dset in tokenized.items():
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')

    if ALL_IN_ONE:
        # ===============IF YOU HAVE ENOUGH MEMORY================
        print("Saving the binary... Please wait this could take a while...")
        dset = dset.with_format('numpy')
        arr = np.concatenate(dset['ids']).astype(np.uint32)
        arr.tofile(filename)
    else:
        # ===============IF YOU DON'T HAVE ENOUGH MEMORY================
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        arr = np.memmap(filename, dtype=np.uint32, mode='w+', shape=(arr_len,))

        idx = 0
        for batch_idx in tqdm(range(BATCH_SIZE), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=BATCH_SIZE, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
    arr.flush()
    print(f"Saved {split} dataset to {filename}")