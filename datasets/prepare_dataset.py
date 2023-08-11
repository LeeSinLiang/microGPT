from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import numpy as np

tokenizer_path="tokenizer/tokenizer.json"
dataset_dir="datasets/"

tokenizer = GPT2TokenizerFast(
    tokenizer_file=tokenizer_path,
    pad_token="[PAD]",
    padding_side="left",
)

dataset = load_dataset("JeanKaddour/minipile")

def process(data):
    inpt = tokenizer(data['text'])
    inpt['input_ids'].append(tokenizer.eos_token_id)
    out = {'input_ids': inpt['input_ids'], 'len': len(inpt['input_ids'])}
    return out

dataset = dataset.map(
        process,
        remove_columns=['text'],
        num_proc=8,
)

for split, dset in dataset.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(dataset_dir, f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 500

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['input_ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()