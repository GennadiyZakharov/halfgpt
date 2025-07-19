
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, seq_len, text):

        # Load GPT2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 has no pad token; use eos
        tokens = self.tokenizer.encode(text)

        # Make sure total tokens are divisible by SEQ_LEN
        total_len = (len(tokens) // seq_len) * seq_len
        tokens = tokens[:total_len]  # truncate
        input_ids = torch.tensor(tokens, dtype=torch.long)
        # Split into chunks of SEQ_LEN
        input_ids = input_ids.view(-1, seq_len)  # shape: (num_sequences, seq_len)
        self.data = input_ids

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.clone(x)
        y[:-1] = x[1:]      # shift left
        y[-1] = self.tokenizer.eos_token_id  # or just copy y[-1] = x[-1]
        return {"input_ids": x, "labels": y}