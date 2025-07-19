#!/usr/bin/env python
import torch
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset
import time
import os

from halfgpt import data_loader, gpt_model


num_epochs = 100
vocab_size = 50257
embed_dim = 768
num_heads = 12
num_layers = 12
max_len = 512
BATCH_SIZE = 8
REPORT_PERIOD = 10

def getwikitext():
    # Load WikiText-2 (raw version)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(dataset["text"])  # long text corpus
    print(f"Full text length: {len(full_text)}")
    return full_text

def main():
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path_global = os.path.join(checkpoint_dir, f"halfgpt.model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text = getwikitext()
    train_dataset = data_loader.GPTDataset(max_len, text)
    print(f"Total {len(train_dataset)} chunks, {max_len} tokens each", )

    model = gpt_model.GPTModel(vocab_size, max_len, embed_dim, num_layers, num_heads).to(device)
    if os.path.exists(checkpoint_path_global):
        model.load_state_dict(torch.load(checkpoint_path_global, weights_only=False))
        print(f"Loaded model checkpoint from {checkpoint_path_global}")
    else:
        print("No checkpoint found, starting from scratch")

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        train_dataloader = data_loader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        n_batches = len(train_dataloader)
        epoch_start_time = time.time()
        for n, batch in enumerate(train_dataloader):  # dataloader yields (input_ids, target_ids) tensors
            inputs = batch["input_ids"].to(device)    # shape (B, L)
            targets = batch["labels"].to(device)      # shape (B, L)
            optimizer.zero_grad()
            logits = model(inputs)                    # shape (B, L, vocab_size)
            # Reshape outputs and targets to compute cross-entropy
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            if n % REPORT_PERIOD == 0:
                #checkpoint_path = os.path.join(checkpoint_dir, f"halfgpt.epoch_{epoch}-{n}.pt")
                #torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), checkpoint_path_global)
                report_period_duration = (time.time() - epoch_start_time)/(n+1)
                print(f"Epoch {epoch}, batch {n}/{n_batches} training loss: {loss.item():.4f}")
                print(f"Duration per {REPORT_PERIOD} batches: {report_period_duration:.2f}s")
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
