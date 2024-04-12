from dataclasses import dataclass
from gpt import GPT
from transformers import GPT2TokenizerFast
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modify the parameters here
max_length = 512
model_path = "models/microGPT.pth"
tokenizer_path = "tokenizer/tokenizer.json"
n_tokens = 1000
temperature = 0.8
top_k = 0
top_p = 0.9

tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)

@dataclass
class GPTConfig:
    n_embd = 768
    vocab_size = len(tokenizer.get_vocab())
    max_length = 512
    n_head = 8
    n_layer = 8
    dropout = 0.0
    training = True
    pad_token = tokenizer.convert_tokens_to_ids('[PAD]')
    
config = GPTConfig
model = GPT(config)

model_stat = torch.load(model_path)
model.load_state_dict(model_stat["model_state_dict"])
model = model.to(device)

# If you train on the original dataset that the model is trained (minipile https://arxiv.org/abs/2304.08442), the model can generate code, stories, dialogues... etc
context = '''Marlene: Good afternoon Houston division, I am so excited to be here with you talking about an exciting quarter for our division. We are so excited to introduce someone who is here with us for the first time. Rachel Ross!
Rachel: Thank you Marlene. In March, I assumed the role of Vice President of Merchandising for the Houston Division. I came from the Michigan Division so the heat and humidity has been quite a change, but being with this division’s team has been so amazing.
Marlene: Rachel, we are glad to have you here and excited about all of the energy you have already brought to the team. First let’s hear from our Division Controller, Akin Akanni, about how we did financially in the Houston Division this quarter.
Akin: Thanks guys, He spoke of how rare it is to receive the amazing level service that he provided in other stores. Thank you Brent for giving our customers highly satisfying service.  We are so proud to have you on our Houston team.
Marlene and Mike: Way to go Brent!
'''
context = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).reshape(1, -1).to(device)
print(
    tokenizer.decode(
        model.generate(
            context, max_tokens_generate=n_tokens, top_k=top_k, top_p=top_p, temperature=temperature
        ).tolist()
    )
)
