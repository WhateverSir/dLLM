# dLLM
Diffusion Large Language Model
## MDLM：研究型dLLM的突破
MDLM（Masked Diffusion Language Models）是扩散语言模型的一种研究实现，其核心是基于掩盖的离散扩散模型，采用紧凑的Rao-Blackwellized目标函数，无需复杂的CTMC理论。
![image](https://github.com/user-attachments/assets/66fc0b8c-4c4f-4b08-96be-4be25a6a9f5c)
## Mercury：商业化dLLM的里程碑
Mercury Coder采用粗到细（coarse-to-fine）的生成方式，与自回归模型的左到右顺序生成不同。其生成速度可达每秒1000个token，在NVIDIA H100 GPU上比传统LLM快5-10倍。早期基准测试显示，其质量可与GPT-4o Mini和Claude 3.5 Haiku相当，同时成本更低。
![image](https://github.com/user-attachments/assets/6c138e5f-b6aa-414c-8930-b4e4fde9fc27)

示例能帮助你理解如何构建和训练一个dLLM。
## 示例关键步骤说明
* 模型构建 ：
TransformerModel 类定义了Transformer模型的架构，包括嵌入层、位置编码、Transformer编码器和全连接层。
PositionalEncoding 类为输入序列添加位置编码，以捕捉序列中的位置信息。
* 训练语料处理 ：
TextDataset 类从文件中读取文本数据，并将其转换为模型可以接受的格式。
create_data_loader 函数创建数据加载器，用于批处理数据。
* 噪声添加 ：
add_noise 函数在输入序列中随机替换一些标记，以增加模型的鲁棒性。
* 训练脚本 ：
训练循环中，模型通过前向传播生成输出，然后通过逆向传播更新参数。
criterion 是交叉熵损失函数，用于计算预测结果与真实标签之间的差异。
optimizer 是AdamW优化器，用于更新模型参数。
* 评估脚本 ：
评估循环中，模型在测试数据上进行前向传播，生成预测结果。
通过比较预测结果与真实标签，计算准确率。
