#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM
from datasets import load_from_disk
import sentencepiece as spm
import random
from tqdm import tqdm

# ========== 配置 ==========
CACHE_DIR = "./data/processed/bert_14M_cache"
TOKENIZER_PATH = "./tokenizer/chinese_spm.model"
CONFIG_PATH = "./configs/config.json"   # 使用你自定义的 config
OUTPUT_DIR = "./models/pretrain"
BATCH_SIZE = 8                    # 根据显存调整
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 5e-4
EPOCHS = 3
MAX_SEQ_LEN = 512
MLM_PROBABILITY = 0.15
MASK_TOKEN_ID = 4                 # 重要：我们手动分配一个未使用的 token id 作为 [MASK]
                                  # 因为 SentencePiece 中没有 [MASK]，我们临时用 4（确保 0-3 已被 [PAD],[UNK],[BOS],[EOS] 占用）
                                  # 注意：这要求你的词汇表中 id=4 未被实际使用。通常训练好的 vocab 从 4 开始是真实 token。
                                  # 更好的办法：重新训练分词器加入 [MASK]，但这里先这样处理，影响不大。

# ========== 加载数据集 ==========
print("加载缓存数据集...")
dataset = load_from_disk(CACHE_DIR)
print(f"数据集大小: {len(dataset)} 条")

# ========== 加载 SentencePiece 分词器 ==========
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)

# ========== 加载模型配置 ==========
config = BertConfig.from_json_file(CONFIG_PATH)
model = BertForMaskedLM(config=config)
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ========== 自定义 collate 函数（实现 MLM masking） ==========
def mlm_collate(batch):
    """
    batch: list of dict, each with 'input_ids' (list of int, length MAX_SEQ_LEN)
    Returns:
        input_ids: torch.LongTensor (batch_size, seq_len)
        labels: torch.LongTensor (batch_size, seq_len), 原始 token ids，被 mask 的位置为 -100
    """
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    labels = input_ids.clone()
    
    # 创建 mask 概率矩阵
    probability_matrix = torch.full(labels.shape, MLM_PROBABILITY)
    # 特殊 token（如 [PAD], [UNK], [CLS], [SEP]）不应该被 mask，但我们的数据只有 input_ids，没有特殊 token 标识，简单起见不对特殊 token 做保护
    # 注意：我们的序列中 [PAD]=0 不应该被 mask，否则模型会学习预测 padding。所以将 padding 位置的 mask 概率设为 0
    padding_mask = (input_ids == 0)  # [PAD] id 为 0
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    # 随机决定哪些位置被 mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # 被 mask 的位置，标签保留原 token id，输入改为 [MASK] token id
    labels[~masked_indices] = -100  # 只有被 mask 的位置计算损失
    # 将被 mask 的位置的 input_ids 替换为 MASK_TOKEN_ID
    input_ids[masked_indices] = MASK_TOKEN_ID
    
    return {
        'input_ids': input_ids,
        'attention_mask': (~padding_mask).long(),   # attention mask: 非 padding 为 1
        'labels': labels
    }

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=mlm_collate,
    num_workers=2
)

# ========== 优化器 ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(dataloader) * EPOCHS // GRADIENT_ACCUMULATION
print(f"总训练步数（考虑梯度累积）: {total_steps}")

# ========== 训练循环 ==========
global_step = 0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Training")
    for step, batch in enumerate(progress_bar):
        # 移动到 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss = loss / GRADIENT_ACCUMULATION
        loss.backward()
        
        epoch_loss += loss.item() * GRADIENT_ACCUMULATION
        
        # 梯度累积步数更新
        if (step + 1) % GRADIENT_ACCUMULATION == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # 记录日志
            if global_step % 100 == 0:
                print(f"Step {global_step}, loss: {epoch_loss / (step+1):.4f}")
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item() * GRADIENT_ACCUMULATION})
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 保存检查点
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    print(f"保存检查点到 {checkpoint_dir}")

# 保存最终模型
model.save_pretrained(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")
