from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch

max_length = 512

def main():
    # Load a dataset from the Hugging Face library
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset['train']['text']

    # Using a pre-trained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Define a function to tokenize the data
    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    # Tokenize the dataset
    train_encodings = tokenize_function(train_texts)

    # Convert the tokenized data into a format suitable for PyTorch.
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])

    train_dataset = TextDataset(train_encodings)

if __name__ == "__main__":
    main()
