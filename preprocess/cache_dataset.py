import os
import shutil

import sentencepiece as spm
from datasets import DatasetDict, load_from_disk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "train-set", "untokenized", "train-formatted.arrow")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "chinese_spm.model")
CACHE_PATH = os.path.join(PROJECT_ROOT, "train-set", "tokenized", "train-formatted.arrow")

MAX_LEN = 128


def tokenize_batch(examples, sp_path, max_len):
    if not hasattr(tokenize_batch, "sp"):
        tokenize_batch.sp = spm.SentencePieceProcessor()
        tokenize_batch.sp.Load(sp_path)

    sp = tokenize_batch.sp
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    max_content_len = max_len - 2

    input_ids_list = []
    for text in examples["text"]:
        content_ids = sp.EncodeAsIds(text or "")[:max_content_len]
        input_ids_list.append([bos_id] + content_ids + [eos_id])

    return {"input_ids": input_ids_list}


def main():
    print(f"Loading raw dataset: {RAW_DATA_PATH}")
    dataset_or_dict = load_from_disk(RAW_DATA_PATH)
    if isinstance(dataset_or_dict, DatasetDict) and "train" in dataset_or_dict:
        dataset = dataset_or_dict["train"]
    else:
        dataset = dataset_or_dict

    print(f"Loaded {len(dataset)} samples")
    worker_count = max(1, (os.cpu_count() or 2) // 2)
    print(
        "Tokenizing with decoder contract: "
        f"BOS + ids[:{MAX_LEN - 2}] + EOS, max_len={MAX_LEN}, workers={worker_count}"
    )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        fn_kwargs={"sp_path": TOKENIZER_PATH, "max_len": MAX_LEN},
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=worker_count,
        writer_batch_size=1000,
        load_from_cache_file=False,
    )

    if os.path.exists(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)

    print(f"Saving tokenized dataset: {CACHE_PATH}")
    tokenized_dataset.save_to_disk(CACHE_PATH)
    print("Preprocess complete")


if __name__ == "__main__":
    main()
