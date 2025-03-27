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
