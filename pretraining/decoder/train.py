import os
import math
import torch
import sys
import sentencepiece as spm
from tqdm import tqdm
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPT2Config, GPT2LMHeadModel, LlamaTokenizer
from trl import SFTTrainer, SFTConfig
import torch.nn as nn

# 解决 OpenMP 冲突（针对 Windows）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 1. 参数与路径配置 (软编码) ====================
WORKSPACE_ROOT = "../../"
TRAIN_DATA_PATH = os.path.join(WORKSPACE_ROOT, "train-set/tokenized/train-formatted.arrow/")
TOKENIZER_PATH = os.path.join(WORKSPACE_ROOT, "tokenizer/chinese_spm.model")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "models/pretrain/decoder")
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "logs/decoder_pretrain")

# 训练超参数
MAX_LEN = 128
BATCH_SIZE = 8  # 考虑到 4060 显存，设为 8 更稳妥
LEARNING_RATE = 1e-4
EPOCHS = 3
SAVE_STEPS = 500
LOGGING_STEPS = 100
WEIGHT_DECAY = 0.01

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_decoder_model(vocab_size):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_LEN,
        n_ctx=MAX_LEN,
        n_embd=384,
        n_layer=6,
        n_head=12,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    return GPT2LMHeadModel(config)

# ==================== 2. 训练主流程 ====================
def main():
    print(f"当前运行环境: {DEVICE}")
    if DEVICE == "cuda":
        print(f"检测到显卡: {torch.cuda.get_device_name(0)}")

    # 1. 加载分词器
    print(f"正在加载分词器: {TOKENIZER_PATH}")
    # 使用 LlamaTokenizer 加载 SentencePiece 模型，并设置特殊 Token
    tokenizer = LlamaTokenizer(vocab_file=TOKENIZER_PATH)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 2
    tokenizer.eos_token_id = 3
    vocab_size = len(tokenizer)

    # 2. 加载数据集
    print(f"正在读取数据集 (千万级索引建立中，请耐心等候...): {TRAIN_DATA_PATH}")
    
    import datasets
    # 彻底关闭数据集库的内部日志和进度条，防止干扰
    datasets.logging.set_verbosity_error() 
    
    # 保持磁盘映射，解决 Windows 卡延迟
    dataset = load_from_disk(TRAIN_DATA_PATH, keep_in_memory=False)
    
    # 自动识别 train 键
    if isinstance(dataset, datasets.DatasetDict):
        train_dataset = dataset['train']
    else:
        train_dataset = dataset

    print(f"--- 磁盘映射完成！ ---")
    print(f"样本总数: {len(train_dataset)}")
    
    # 关键测试：取一条数据看响应时间
    print("正在验证数据读取响应...")
    _ = train_dataset[0]
    print("数据读取正常。")

    # 3. 初始化模型
    print("正在构建模型架构...")
    model = get_decoder_model(vocab_size)
    model.to(DEVICE)
    print(f"模型已移至: {DEVICE}")

    # 4. 配置训练参数 (使用 SFTConfig 解决警告)
    print("正在配置训练参数 (禁用冗余扫描)...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_steps=5,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_dir=LOGS_DIR,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=False,
        dataloader_pin_memory=False,
        full_determinism=False,
        # SFTConfig 特有参数
        max_seq_length=MAX_LEN,
        packing=True,
        dataset_text_field="input_ids", # 即使 dataset_text_field 为 None，内部逻辑有时也需要指定
    )

    # 5. 训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,             # 必须显式传入 tokenizer 解决 OSError
    )

    print("--- 准备就绪，即将开始训练。注意：由于数据量巨大，启动前会有 30-60 秒静默期 ---")

    
    # 断点续传逻辑
    resume_from_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if dirs:
            try:
                dirs.sort(key=lambda x: int(x.split("-")[-1]))
                resume_from_checkpoint = os.path.join(OUTPUT_DIR, dirs[-1])
                print(f"检测到历史存档，将从 {resume_from_checkpoint} 恢复训练...")
            except: pass

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("训练全部完成！")

if __name__ == "__main__":
    main()

