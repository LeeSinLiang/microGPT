from gpt import GPT
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from transformers import GPT2TokenizerFast
from config import GPTConfig
import numpy as np
import math
import torch
import os
import wandb
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

max_length = 512
tokenizer_path="tokenizer/tokenizer.json"
tokenizer = GPT2TokenizerFast(
    tokenizer_file=tokenizer_path,
    pad_token="[PAD]",
    padding_side="right",
    model_max_length=max_length,
)

# hyperparameters
config_file="config/config.json"

# model config
max_length = 512
batch_size = 12
num_accumulation_steps = 5 * 8
n_steps = 600000
epochs = 1
grad_norm_clip = 1.0
checkpoint_interval = 10000
eval_interval = 2000
eval_iters = 100
vocab_size = len(tokenizer.get_vocab())
save_directory = "models"
# adamw optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True
config_lr = {
    'warmup_steps': 2000,
    'lr_decay_steps': n_steps,
    'min_lr': 6e-5,
    'lr': learning_rate
}

# checkpoints
checkpoint=False
model_path=""

# wandb
wandb_log=True
project="GPT Training"
name=""
resume=False
id=None

# dataset
dataset = "memmap"
data_dir = 'datasets'

if dataset == "memmap":
    train_data = np.memmap(os.path.join(
    data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(
        data_dir, 'validation.bin'), dtype=np.uint16, mode='r')
elif dataset == "huggingface":
    from datasets import load_from_disk

    num_workers=4
    train_data = load_from_disk("datasets/train")
    val_data = load_from_disk("datasets/validation")
    train_data.set_format(type="torch", output_all_columns=True)
    val_data.set_format(type="torch", output_all_columns=True)
    dataloader = DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataloader_val = DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    train_data = iter(dataloader)
    val_data = iter(dataloader_val)
elif dataset == "torch_dataset":
    from dataset import CorpusDataset
    from utils import pad_collate
    from sklearn.model_selection import train_test_split
    
    # parameters
    file_path = ""
    column_text = ""
    with tqdm.tqdm() as bar:
        bar.set_description('Reading CSV')
        data = pd.read_csv(file_path,
                           skiprows=lambda x: bar.update(1) and False)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    train_data, val_data = train_test_split(data, train_size=0.85)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    data = {"train": train_data, "val": val_data}

    datatrain = CorpusDataset(data["train"][column_text])
    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: pad_collate(x, tokenizer),
    )
    datatest = CorpusDataset(data["val"][column_text])
    dataloader_val = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: pad_collate(x, tokenizer),
    )
    train_data = iter(dataloader)
    val_data = iter(dataloader_val)
else:
    print("Invalid dataset.")

def get_batch(split):
    if dataset == "memmap":
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - max_length, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i:i+max_length]).astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy((data[i+1:i+1+max_length]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
    else:
        inpt = next(train_data) if split == 'train' else next(val_data)
        x = inpt[:, :max_length]
        y = inpt[:, 1:max_length+1]
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters):
    model.eval()
    out = {}
    with tqdm(total=eval_iters * 2, unit="batch", position=1, leave=True) as eval_steps:
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            k = 0
            while k < eval_iters:
                xBatch, yBatch = get_batch(split)
                _, loss = model(xBatch, target=yBatch)
                losses[k] = loss.item()
                k += 1
                eval_steps.set_description('Loss Estimation')
                eval_steps.update(1)
                eval_steps.set_postfix(
                    step_loss=loss.item(), learning_rate=lr, target=split)
            out[split] = losses.mean()
        eval_steps.clear()
    model.train()
    return out


def get_lr_cosine_warmup(config_lr, steps):
    if steps < config_lr['warmup_steps']:
        return config_lr['lr'] * steps / config_lr['warmup_steps']
    if steps > config_lr['lr_decay_steps']:
        return config_lr['min_lr']
    decay_ratio = (steps - config_lr['warmup_steps']) / \
        (config_lr['lr_decay_steps'] - config_lr['warmup_steps'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config_lr['min_lr'] + coeff * (config_lr['lr'] - config_lr['min_lr'])


if __name__ == '__main__':
    with open(config_file, 'r') as f:
        config = json.load(f)

    config = GPTConfig(**config)
    print(
        f'Max length: {config.max_length} Vocab Size: {config.vocab_size}')
    model = GPT(config)
    model = model.to(device)
    step = 0
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type)
    
    if checkpoint:
        model_stat = torch.load(model_path)
        model.load_state_dict(model_stat["model_state_dict"])
        step = model_stat['step']
        optimizer.load_state_dict(model_stat["optimizer_state_dict"])

    scaler = torch.cuda.amp.GradScaler()
    if wandb_log:
        wandb.init(
            project=project,
            resume=resume,
            id=id,
            name=name,
            config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": optimizer.param_groups[0]["lr"],
                "loss_fn": "crossentropyloss",
            },
        )

    print('Device: ', device)
    print(tokenizer.decode(tokenizer.encode(
        "---------------GPT Model----------------")))

    # AMP & Gradient Accumulation to prevent CUDA Out of Memory Error
    scaler = torch.cuda.amp.GradScaler()

    with tqdm(total=n_steps, unit="batch", position=0, leave=True) as t_steps:
        t_steps.update(step)
        while step < n_steps:
            xBatch, yBatch = get_batch('train')

            lr = get_lr_cosine_warmup(
                config_lr, step + 1) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with torch.autocast(device_type=device_type):
                logits, loss = model(xBatch, target=yBatch)

            scaler.scale(loss / num_accumulation_steps).backward()
            t_steps.update(1)
            t_steps.set_postfix(step_loss=loss.item(), learning_rate=lr)
            if ((step + 1) % num_accumulation_steps == 0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (step % (checkpoint_interval) == 0):
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"{save_directory}/microGPT-step-{step}.pth",
                )
            if step % (eval_interval) == 0:
                losses = estimate_loss(
                    model,
                    eval_iters=eval_iters,
                )
                print(
                    f"Current step {step}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, lr: {lr}"
                )
                if wandb_log:
                    metrics = {
                        "train/train_loss": losses["train"],
                        "train/steps": step,
                        "validation/val_loss": losses["val"],
                        "validation/steps": step,
                        "models/lr": lr
                    }
                    wandb.log(metrics)
            step += 1

    losses = estimate_loss(
        model,
        eval_iters=eval_iters,
    )
    print(
        f"Final train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    context = torch.tensor(tokenizer.encode(
        "How are you?\n"), dtype=torch.long, device=device).reshape(1, -1)
    print(
        tokenizer.decode(
            model.generate(
                context, max_tokens_generate=5000, top_k=0, top_p=0.9, temperature=0.8
            ).tolist()
        )
    )
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{save_directory}/microGPT.pth"
    )
