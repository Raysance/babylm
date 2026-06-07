import os
import sentencepiece as spm
from datasets import load_from_disk, Dataset
from tqdm import tqdm

# 配置路径
RAW_DATA_PATH = '../train-set/untokenized/train-formatted.arrow/'                # 原始数据集目录
TOKENIZER_PATH = '../tokenizer/chinese_spm.model'
CACHE_PATH = '../train-set/tokenized/train-formatted.arrow/'

# 3. 批量处理函数
def tokenize_batch(examples, sp_path):
    # 使用静态变量在每个进程内缓存分词器，避免重复加载和 pickling 问题
    if not hasattr(tokenize_batch, "sp"):
        import sentencepiece as spm
        tokenize_batch.sp = spm.SentencePieceProcessor()
        tokenize_batch.sp.Load(sp_path)
    
    sp = tokenize_batch.sp
    input_ids_list = []
    
    for text in examples["text"]:
        # 编码
        raw_ids = sp.EncodeAsIds(text)
        # 视情况加上结束符 EOS 
        raw_ids.append(sp.eos_id()) 
        input_ids_list.append(raw_ids)
        
    return {"input_ids": input_ids_list}

def main():
    # 1. 加载原始数据
    print("正在从磁盘加载原始数据...")
    ds = load_from_disk(RAW_DATA_PATH)
    if isinstance(ds, dict) and 'train' in ds:
        dataset = ds['train']
    else:
        dataset = ds

    print(f"成功加载数据集，总样本数: {len(dataset)}")

    # 4. 开始处理
    # 针对 MemoryError 的优化策略：
    # 1. 减小 num_proc，避免过多元数据竞争内存
    # 2. 减小 batch_size，降低单个进程的峰值内存
    # 3. 设置 writer_batch_size，控制 Arrow 写入时的缓冲区大小
    
    recommended_procs = max(1, os.cpu_count() // 2) # 使用一半的核心，留出内存余量
    print(f"开始 Tokenization (使用进程数: {recommended_procs}, 批大小: 1000)...")
    
    tokenized_dataset = dataset.map(
        tokenize_batch,
        fn_kwargs={"sp_path": TOKENIZER_PATH}, 
        batched=True,
        batch_size=1000, 
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=recommended_procs,
        writer_batch_size=1000, # 关键：限制写入时的内存占用
        load_from_cache_file=True
    )

    # 5. 保存到磁盘
    print(f"处理完成，正在保存至 {CACHE_PATH}...")
    tokenized_dataset.save_to_disk(CACHE_PATH)
    print("预处理全部完成！")

if __name__ == "__main__":
    main()
