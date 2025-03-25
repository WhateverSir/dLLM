import os
import torch
import random
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask

def create_data_loader(tokenizer, file_path, batch_size=32, max_len=128):
    dataset = TextDataset(tokenizer, file_path, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def add_noise(input_ids, noise_rate=0.1):
    noisy_input_ids = input_ids.clone()
    mask = (torch.rand_like(input_ids) < noise_rate).float()
    noisy_input_ids = (noisy_input_ids * (1 - mask)) + (torch.randint(0, tokenizer.vocab_size, input_ids.shape) * mask)
    return noisy_input_ids.long()
