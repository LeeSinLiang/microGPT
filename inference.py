import json
import torch
from gpt import *
from config import GPTConfig
from utils import load_model, load_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters (Edit here):
n_tokens = 1000
temperature = 0.8
top_k = 0
top_p = 0.9
model_file = 'models/recipe-0.7157.pth'

with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = GPTConfig(**config)

model = load_model(model_file, cfg)
char_to_idx, idx_to_char = load_tokenizer('config/vocab.json')
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda idx: ''.join([idx_to_char[str(i)] for i in idx])

# Edit input here
context = "Title: Cheesy Chicken Salad With Pepper And Grated Parmesan Cheese"

with torch.no_grad():
	context = torch.tensor(encode(context), dtype=torch.long, device=device).reshape(1, -1)
	response = decode(model.generate(context, max_tokens_generate=n_tokens, temperature=temperature, top_k=top_k, top_p=top_p).tolist())
	print(response)
