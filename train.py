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
