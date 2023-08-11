import math
import torch
import inspect
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from utils import top_k_top_p_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class HeadAttention(nn.Module):
    def __init__(self, head_size, embedding_dim, max_length, pDropout=0.2):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        # Decoder block: prevent future tokens from talking to current tokens. Only past tokens can talk to current tokens.
        self.register_buffer('tril', torch.tril(
            torch.ones(max_length, max_length)))
        self.head_size = head_size
        self.dropout = nn.Dropout(pDropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        # change to self.head_size # scaled attention (normilization) by dividing it
        qK = self.query(x) @ self.key(x).transpose(-2, -1) * \
            (1.0 / math.sqrt(self.head_size))
        if attention_mask is not None:  # doesnt work, rows of -inf -> nan in cross entropy loss
            qK = attention_mask.unsqueeze(2) * qK
            qK = qK.masked_fill(qK == 0, float('-inf'))
        qK = qK.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        qK = F.softmax(qK, dim=-1)
        qK = self.dropout(qK)
        qK = qK.masked_fill(torch.isnan(qK), 0)
        out = qK @ self.value(x)
        return out


class MultiHeadAttention(nn.Module):
    # mulitple heads of self attention in parallel

    def __init__(self, num_heads, head_size, n_emb, max_length, pDropout):
        super().__init__()
        # self.heads = nn.ModuleList([HeadAttention(head_size, embedding_dim, max_length) for _ in range(num_heads)])
        self.heads = nn.ModuleList(
            [HeadAttention(head_size, n_emb, max_length) for _ in range(num_heads)])
        # linear transformation of self attention
        self.projection = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(pDropout)

    def forward(self, x, attention_mask):
        out = torch.cat([h(x, attention_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_training = config.training
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print("Flash Attention requires PyTorch >= 2.0. Using normal attention.")
            self.register_buffer("bias", torch.tril(torch.ones(
                config.max_length, config.max_length)).view(1, 1, config.max_length, config.max_length))

    def forward(self, x, attn_mask):
        B, T, C = x.shape

        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nhead, T, hsize)
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = torch.ones(B, T, T).tril(
                diagonal=0).to(device) * attn_mask.unsqueeze(1)
            attn_mask = attn_mask.masked_fill(
                attn_mask == 0, -float('inf')).unsqueeze(1).to(query.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0) # dirty workaround
        if self.flash:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.is_training else 0, is_causal=True)
        else:
            # scaled attention (normilization) by dividing it
            if attn_mask is None:
                attn_mask = torch.ones(B, T, T).tril(
                    diagonal=0).unsqueeze(1)  # (B, 1, T, T)
            qK = query(x) @ key(x).transpose(-2, -1) * \
                (1.0 / math.sqrt(query.shape[-1]))
            qK = qK + attn_mask
            qK = F.softmax(qK, dim=-1)
            qK = self.attn_dropout(qK)
            qK = qK.masked_fill(torch.isnan(qK), 0)
            out = qK @ value(x)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.out(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # i.e. 4 heads of 8 dimensional self-attention, which concatenates to 32 (embedding_dim)
        self.sa_head = CausalSelfAttention(config)
        self.feed_fwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)  # pre-normalization
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, attn_mask):
        # x + self... is to fork the computation outside and join back (skip connection)
        # token communication each other
        x = x + self.sa_head(self.ln1(x), attn_mask)
        x = x + self.feed_fwd(self.ln2(x))  # token indiviually think
        return x, attn_mask


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embedding = nn.Embedding(
            config.max_length, config.n_embd)  # to know my tokens location
        self.blocks = MultiInputSequential(
            *[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        # original is embedding_dim
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.max_length = config.max_length
        self.pad_token = config.pad_token
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, attention_mask=None, target=None, ignore=False):
        B, T = idx.shape

        tok_emb = self.embedding_table(idx)  # (B, T, C)
        # torch.arange(T) -> [0, 1,... T] -> (T, C)
        pos_emb = self.positional_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, C)
        x, _ = self.blocks(x, attention_mask)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(-1)  # (B*T)
            if attention_mask is not None or ignore is True:
                loss = F.cross_entropy(
                    logits, target, ignore_index=self.pad_token)  # ignore pad_token
            else:
                loss = F.cross_entropy(logits, target)
        return (logits, loss)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def generate(self, idx: Tensor, max_tokens_generate: int, temperature, top_k, top_p, ):
        # idx (B, T)
        for _ in range(max_tokens_generate):
            # crop it to get latest <max_length> tokens since pos_emb only has max_length size
            idx_condition = idx[:, -self.max_length:]
            logits, loss = self.forward(idx_condition)
            # get the last character logits to predict the next character, (B, C)
            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if (idx_next.item() == 0):
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx.view(idx.shape[1], )
