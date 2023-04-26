import argparse
import json

parser = argparse.ArgumentParser(description='Prepare vocab.json file')
parser.add_argument('text_file', type=str, help='File location of text_file')
parser.add_argument('vocab_file', type=str, help='File location of vocab')
args = parser.parse_args()

with open(args.text_file, 'r', encoding="UTF-8") as f:
	text = f.read()

vocabs = sorted(list(set(text)))
vocab_size = len(vocabs)
print("Size: %s | Vocabs: %s" % (vocab_size, vocabs))

char_to_idx = {char:idx for idx, char in enumerate(vocabs)}
idx_to_char = {idx:char for idx, char in enumerate(vocabs)}

vocab_dict = {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}
with open(args.vocab_file, "w") as f:
    json.dump(vocab_dict, f)
