import torch
import json
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch(batch_size, data, max_length):
	ix = torch.randint(len(data)-max_length, (batch_size, ))
	x = torch.stack([data[i:i+max_length] for i in ix])
	y = torch.stack([data[i+1:i+max_length+1] for i in ix])
	x,y = x.to(device), y.to(device)
	return (x, y)

@torch.no_grad()
def eval_loss(data, model, batch_size, max_length, eval_iters):
	model.eval()
	out = {}
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(batch_size, data[split], max_length)
			X, Y = X.to(device), Y.to(device)
			_, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

def top_k_top_p_filter(logits, top_k:int=0, top_p:float=1.0):
	if top_k > 0:
		filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
		logits[logits < filter[:, [-1]]] = float('-inf')
	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
		filter = cumulative_probs > top_p
		filter[..., 1:] = filter[..., :-1].clone() # shift right by 1 since filter includes the first index that exceeds top_p
		filter[..., 0] = 0

		# convert to original indexing
		indices_to_remove = filter.scatter(1, sorted_indices, filter)
		logits[indices_to_remove] = float('-inf')
	return logits

def load_tokenizer(vocab_file):
	with open(vocab_file, 'r') as f:
		data = json.load(f)

	char_to_idx = data['char_to_idx']
	idx_to_char = data['idx_to_char']

	return char_to_idx, idx_to_char
