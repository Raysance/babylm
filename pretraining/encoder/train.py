import math
import os
import sys

import sentencepiece as spm
import torch
from datasets import DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hf_utils import save_model_and_tokenizer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DATA_PATH = os.path.join(WORKSPACE_ROOT, "train-set", "tokenized", "train-formatted.arrow")
TOKENIZER_PATH = os.path.join(WORKSPACE_ROOT, "tokenizer", "chinese_spm.model")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "models", "pretrain", "encoder")
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "logs", "encoder_pretrain")

MAX_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 3
SAVE_STEPS = 25000
LOGGING_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
MLM_PROBABILITY = 0.15

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
MASK_TOKEN_ID = 4
VOCAB_SIZE = 30000


def get_encoder_model(vocab_size):
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=MAX_LEN,
        type_vocab_size=2,
        pad_token_id=PAD_TOKEN_ID,
    )
    return BertForMaskedLM(config)


def mlm_collate(batch):
    input_ids = [
        torch.tensor(item["input_ids"][:MAX_LEN], dtype=torch.long)
        for item in batch
    ]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask = input_ids.ne(PAD_TOKEN_ID).long()
    labels = input_ids.clone()

    probability_matrix = torch.full(labels.shape, MLM_PROBABILITY)
    special_tokens_mask = (
        input_ids.eq(PAD_TOKEN_ID)
        | input_ids.eq(UNK_TOKEN_ID)
        | input_ids.eq(BOS_TOKEN_ID)
        | input_ids.eq(EOS_TOKEN_ID)
    )
    probability_matrix.masked_fill_(special_tokens_mask, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[replace_with_mask] = MASK_TOKEN_ID

    replace_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~replace_with_mask
    )
    random_words = torch.randint(MASK_TOKEN_ID + 1, VOCAB_SIZE, labels.shape, dtype=torch.long)
    input_ids[replace_with_random] = random_words[replace_with_random]

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


def main():
    is_main_process = int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

    if is_main_process:
        print(f"Encoder training start. GPU count: {torch.cuda.device_count()}")
        print(f"Loading tokenizer: {TOKENIZER_PATH}")

    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_size = sp.GetPieceSize()
    global VOCAB_SIZE
    VOCAB_SIZE = vocab_size
    mask_token_id = sp.piece_to_id("[MASK]")
    if mask_token_id != MASK_TOKEN_ID:
        raise ValueError(f"Expected [MASK] id {MASK_TOKEN_ID}, got {mask_token_id}")

    if is_main_process:
        print(f"Loading tokenized dataset: {TRAIN_DATA_PATH}")

    dataset_or_dict = load_from_disk(TRAIN_DATA_PATH, keep_in_memory=False)
    if isinstance(dataset_or_dict, DatasetDict):
        train_dataset = dataset_or_dict["train"]
    else:
        train_dataset = dataset_or_dict

    if is_main_process:
        print(f"Dataset rows: {len(train_dataset)}")

    model = get_encoder_model(vocab_size)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=mlm_collate,
        num_workers=0,
        pin_memory=False,
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    start_step = 0
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint is not None:
        start_step, checkpoint_dir = latest_checkpoint
        if is_main_process:
            print(f"Resuming from checkpoint: {checkpoint_dir}")
        model = BertForMaskedLM.from_pretrained(checkpoint_dir).to(device)
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

    for _epoch in range(EPOCHS):
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
        save_model_and_tokenizer(model, TOKENIZER_PATH, os.path.join(OUTPUT_DIR, "final"))
        print("Encoder training complete.")


if __name__ == "__main__":
    main()
