
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# Config
SEQ_LEN = 500
BATCH_SIZE = 16

class GPTDataset(Dataset):
    def __init__(self, text):

        # Load GPT2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 has no pad token; use eos

        tokens = self.tokenizer.encode(text)
        print(f"Total tokens: {len(tokens)}")

        # Make sure total tokens are divisible by SEQ_LEN
        total_len = (len(tokens) // SEQ_LEN) * SEQ_LEN
        tokens = tokens[:total_len]  # truncate
        print(f"Total tokens after truncation: {len(tokens)}")

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Split into chunks of SEQ_LEN
        input_ids = input_ids.view(-1, SEQ_LEN)  # shape: (num_sequences, seq_len)
        print(f"Number of chunks: {input_ids.size(0)}")
        self.data = input_ids

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.clone(x)
        y[:-1] = x[1:]      # shift left
        y[-1] = self.tokenizer.eos_token_id  # or just copy y[-1] = x[-1]
        return {"input_ids": x, "labels": y}