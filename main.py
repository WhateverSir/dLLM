# 1. 模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, mask=src_mask, key_padding_mask=src_key_padding_mask)
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# 2. 训练语料处理
import os
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
# 3. 噪声添加
def add_noise(input_ids, noise_rate=0.1):
    noisy_input_ids = input_ids.clone()
    mask = (torch.rand_like(input_ids) < noise_rate).float()
    noisy_input_ids = (noisy_input_ids * (1 - mask)) + (torch.randint(0, tokenizer.vocab_size, input_ids.shape) * mask)
    return noisy_input_ids.long()
# 4. 训练脚本
import math
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

# 定义超参数
vocab_size = 30522  # BERT词汇表大小
d_model = 768
nhead = 12
num_layers = 6
dim_feedforward = 3072
dropout = 0.1
batch_size = 32
max_len = 128
learning_rate = 1e-4
num_epochs = 10

# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建数据加载器
train_data_loader = create_data_loader(tokenizer, 'train.txt', batch_size, max_len)

# 初始化模型、损失函数和优化器
model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask) in enumerate(train_data_loader):
        optimizer.zero_grad()

        # 添加噪声
        noisy_input_ids = add_noise(input_ids)

        # 前向传播
        outputs = model(noisy_input_ids, src_mask=attention_mask)
        logits = outputs.view(-1, vocab_size)
        labels = input_ids.view(-1)

        # 计算损失
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # 逆向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_data_loader)}')
# 5. 评估脚本
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

# 定义超参数
vocab_size = 30522  # BERT词汇表大小
d_model = 768
nhead = 12
num_layers = 6
dim_feedforward = 3072
dropout = 0.1
batch_size = 32
max_len = 128

# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建数据加载器
test_data_loader = create_data_loader(tokenizer, 'test.txt', batch_size, max_len)

# 初始化模型
model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 评估循环
correct = 0
total = 0
with torch.no_grad():
    for input_ids, attention_mask in test_data_loader:
        outputs = model(input_ids, src_mask=attention_mask)
        logits = outputs.view(-1, vocab_size)
        _, predicted = torch.max(logits, 1)
        labels = input_ids.view(-1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
