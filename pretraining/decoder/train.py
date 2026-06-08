import os
import math
import torch
import sys
import sentencepiece as spm
from tqdm import tqdm
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hf_utils import save_model_and_tokenizer

# 解决 OpenMP 冲突（针对 Windows）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 1. 参数与路径配置 (软编码) ====================
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DATA_PATH = os.path.join(WORKSPACE_ROOT, "train-set/tokenized/train-formatted.arrow/")
TOKENIZER_PATH = os.path.join(WORKSPACE_ROOT, "tokenizer/chinese_spm.model")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "models/pretrain/decoder")
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "logs/decoder_pretrain")

# 训练超参数
MAX_LEN = 128
BATCH_SIZE = 32  # 4090 显存充足，调大 Batch Size 提升吞吐量
LEARNING_RATE = 2e-4 # 随着 Batch Size 变大，学习率适当调高
EPOCHS = 3
SAVE_STEPS = 25000
LOGGING_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

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


def causal_lm_collate(batch):
    input_ids = [
        torch.tensor(item["input_ids"][:MAX_LEN], dtype=torch.long)
        for item in batch
    ]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.clone()
    labels[input_ids == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for dirname in os.listdir(output_dir):
        if not dirname.startswith("checkpoint-"):
            continue
        try:
            step = int(dirname.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, os.path.join(output_dir, dirname)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1]

# ==================== 2. 训练主流程 ====================
def main():
    # 获取分布式环境状态
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]

    if is_main_process:
        print(f"主进程启动。检测到 GPU 总数: {torch.cuda.device_count()}")

    # 1. 加载分词器
    if is_main_process:
        print(f"正在加载分词器: {TOKENIZER_PATH}")
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_size = sp.GetPieceSize()

    # 2. 加载数据集
    if is_main_process:
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

    if is_main_process:
        print(f"--- 磁盘映射完成！ ---")
        print(f"样本总数: {len(train_dataset)}")
    
    # 3. 初始化模型
    if is_main_process:
        print("正在构建模型架构...")
    model = get_decoder_model(vocab_size)
    # 不再手动移至 DEVICE，由 Trainer 处理分布式分配

    # 4. 配置训练组件
    if is_main_process:
        print("正在配置训练参数...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=causal_lm_collate,
        num_workers=0,
        pin_memory=False,
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if is_main_process:
        print("--- 准备就绪，即将开始训练。注意：由于数据量巨大，启动前会有 30-60 秒静默期 ---")

    
    start_step = 0
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint is not None:
        start_step, checkpoint_dir = latest_checkpoint
        if is_main_process:
            print(f"检测到历史存档，将从 {checkpoint_dir} 恢复训练...")
        model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
    else:
        model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if latest_checkpoint is not None:
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    model.train()
    global_step = start_step
    skipped_steps = 0
    total_steps = math.ceil(len(dataloader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS)
    progress_bar = tqdm(total=total_steps, initial=start_step, disable=not is_main_process)

    for epoch in range(EPOCHS):
        for batch in dataloader:
            if global_step >= total_steps:
                break

            if skipped_steps < start_step:
                skipped_steps += 1
                continue

            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(**batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()

            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if is_main_process:
                progress_bar.update(1)
                if global_step % LOGGING_STEPS == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

            if global_step % SAVE_STEPS == 0:
                checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                save_model_and_tokenizer(model, TOKENIZER_PATH, checkpoint_dir)
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

    if is_main_process:
        progress_bar.close()
    
    if is_main_process:
        save_model_and_tokenizer(model, TOKENIZER_PATH, os.path.join(OUTPUT_DIR, "final"))
        print("训练全部完成！")

if __name__ == "__main__":
    main()

