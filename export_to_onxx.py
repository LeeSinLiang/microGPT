import json
import torch
from gpt import *
from config import GPTConfig
from utils import load_model, load_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters (Edit here):
n_tokens = 100
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

context = "Title: Cheesy Chicken Salad With Pepper And Grated Parmesan Cheese"


# with torch.no_grad():
# 	context = torch.tensor(encode(context), dtype=torch.long, device=device).reshape(1, -1)
# 	response = decode(model.generate(context, max_tokens_generate=n_tokens, temperature=temperature, top_k=top_k, top_p=top_p).tolist())
# 	# print(response)

# print(context.size())
context = torch.randint(0,97, (64, 256))
print(context.shape)
torch.onnx.export(model,                     # model being run
                (context, ),           # model input (or a tuple for multiple inputs)
                "models/microGPT.onnx",      # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = 'input',   # the model's input names
                output_names = 'output', # the model's output names
                dynamic_axes={'input' : {1 : 'batch_size',  1: 'max_length'},    # (64, 256)
                              'output' : {2 : 'logits'}}) # 256, 97
