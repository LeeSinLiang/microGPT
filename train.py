import torch
import argparse
import json

from torch.utils.data import DataLoader

from config import GPTConfig
from dataset import CorpusDataset
from gpt import GPT
from utils import eval_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(vocab_file):
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    char_to_idx = data['char_to_idx']
    idx_to_char = data['idx_to_char']

    return char_to_idx, idx_to_char


def load_data(text_file, encode):
	with open(text_file, 'r', encoding="UTF-8") as f:
		text = f.read()
	data = torch.tensor(encode(text), dtype=torch.long)
	n = int(0.85*len(data))
	data = {'train': data[:n], 'val': data[n:]}
	return data


def load_model(config):

    model = GPT(config.vocab_size, config.max_length,
                config.n_emb, config.n_head, config.n_layer)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    return model, optimizer


parser = argparse.ArgumentParser(description='GPT Configuration')
parser.add_argument('config_file', type=str,
                    help='File location of configuration json file')
parser.add_argument('vocab_file', type=str,
                    help='File location of vocab json file')
parser.add_argument('text_file', type=str, help='File location of text_file')
parser.add_argument('save_model_dir', type=str,
                    help='Directory path to save model')

args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = json.load(f)

config = GPTConfig(**config)
char_to_idx, idx_to_char = load_tokenizer(args.vocab_file)

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda idx: ''.join([idx_to_char[str(i)] for i in idx])

data = load_data(args.text_file, encode)
corpusText = CorpusDataset(data['train'], config.max_length)
dataloader = DataLoader(corpusText, batch_size=config.batch_size, shuffle=True)
data_iter = iter(dataloader)
print("Data loaded successfully!")

# sanity check
print(decode(encode('---------------GPT Model----------------')))

model, optimizer = load_model(config)
print("Model loaded successfully!")

scaler = torch.cuda.amp.GradScaler()

for step in range(config.n_steps):
	batch = next(data_iter)
	with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
		x,y = batch
		xBatch, yBatch = x.to(device), y.to(device)
		logits, loss = model(xBatch, yBatch)

	optimizer.zero_grad(set_to_none=True)
	scaler.scale(loss).backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)
	scaler.step(optimizer)
	scaler.update()

	if (step % (config.eval_interval) == 0):
		losses = eval_loss(data, model, config.batch_size,
				config.max_length, eval_iters=config.eval_iters)
		print(
			f"Current step {step}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

losses = eval_loss(data, model, config.batch_size,
                   config.max_length, eval_iters=config.eval_iters)
print(
    f"Final train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
model.eval()
print(decode(model.generate(context, max_tokens_generate=500, top_k=0, top_p=0.9, temperature=0.8).tolist()))
torch.save(model.state_dict(),
           f"{args.save_model_dir}/model-{losses['val']:.4f}.pth")
