#!/usr/bin/env python

import os

import torch
from transformers import GPT2Tokenizer

# --- Config ---
MAX_NEW_TOKENS = 50  # Number of tokens to generate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Once upon a time"

# --- Load tokenizer ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# --- Load model ---
from halfgpt import gpt_model

# Define same hyperparameters as during training
num_epochs = 100
vocab_size = tokenizer.vocab_size
embed_dim = 768
num_heads = 12
num_layers = 12
max_len = 512

model = gpt_model.GPTModel(vocab_size, max_len, embed_dim, num_layers, num_heads)

checkpoint_dir = "checkpoints"
checkpoint_path_global = os.path.join(checkpoint_dir, f"halfgpt.model.pt")

if os.path.exists(checkpoint_path_global):
    model.load_state_dict(torch.load(checkpoint_path_global, weights_only=False))
    print(f"Loaded model checkpoint from {checkpoint_path_global}")
else:
    print("No checkpoint found, starting from scratch")

model.to(DEVICE)
model.eval()

# --- Encode prompt ---
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)  # shape: (1, prompt_len)
generated = input_ids

# --- Generate tokens ---
print("\n ===Generating output: === \n")
for _ in range(MAX_NEW_TOKENS):
    # Use only the last `max_len` tokens as context if needed
    input_trunc = generated[:, -max_len:]

    with torch.no_grad():
        logits = model(input_trunc)  # shape: (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # take logits at last position
        probs = torch.softmax(next_token_logits, dim=-1)

    # Option 1: Greedy decoding (most likely token)
    # next_token = torch.argmax(probs, dim=-1, keepdim=True)

    # Option 2: Sampling
    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

    generated = torch.cat((generated, next_token), dim=1)

# --- Decode generated tokens ---
output_text = tokenizer.decode(generated[0].tolist())
print("\n ===Generated output: === \n")
print(output_text)
