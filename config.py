from dataclasses import dataclass

@dataclass
class GPTConfig:
	batch_size:int
	max_length:int
	lr:float
	n_steps:int
	eval_interval:int
	eval_iters:int
	n_emb:int
	n_head:int
	n_layer:int
	pDropout:int
	grad_norm_clip:float
	vocab_size:int
