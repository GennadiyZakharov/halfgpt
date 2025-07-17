#!/usr/bin/env python
import torch
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset
import time

from halfgpt import data_loader, gpt_model

#max_length = 512
num_epochs = 100
vocab_size = 50257
embed_dim = 768
num_heads = 12
num_layers = 12
max_len = 512

def getwikitext():
    # Load WikiText-2 (raw version)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(dataset["text"])  # long text corpus
    print(f"Full text length: {len(full_text)}")
    return full_text

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text = getwikitext()
    train_dataset = data_loader.GPTDataset(text)
    print("Total chunks of tokens", len(train_dataset))
    train_dataloader = data_loader.DataLoader(train_dataset, batch_size=data_loader.BATCH_SIZE, shuffle=True)

    model = gpt_model.GPTModel(vocab_size, max_len, embed_dim, num_layers, num_heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        epoch_start_time = time.time()
        for n, batch in enumerate(train_dataloader):  # dataloader yields (input_ids, target_ids) tensors
            inputs = batch["input_ids"].to(device)    # shape (B, L)
            targets = batch["labels"].to(device)      # shape (B, L)
            optimizer.zero_grad()
            logits = model(inputs)                    # shape (B, L, vocab_size)
            # Reshape outputs and targets to compute cross-entropy
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            print(f"Batch {n} training loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch}, training loss: {loss.item():.4f}, duration: {epoch_duration:.2f} seconds")
        

if __name__ == "__main__":
    main()
