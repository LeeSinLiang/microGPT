from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

# parameters
num_proc=4 # Number of processes when downloading and generating the dataset locally
vocab_size=16384
file_path="tokenizer/tokenizer.json"

def preprocess_text(data):
    data['text'] = data['text'] + '<|endoftext|>'
    return data

def get_training_corpus(dataset):
    for i in range(0, len(dataset), 10000):
        yield dataset[i : i + 100]["text"]

dataset = load_dataset("JeanKaddour/minipile")["train"]
dataset = dataset.map(preprocess_text, num_proc=num_proc)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "[PAD]"])
tokenizer.model = models.BPE()
tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.save(file_path)
print(tokenizer.decode(tokenizer.encode("Done.").ids))