from dataclasses import dataclass

@dataclass
class GPTConfig:
    n_embd:int
    vocab_size:int
    max_length:int
    n_head:int
    n_layer:int
    dropout:float
    training:bool
    pad_token:int
