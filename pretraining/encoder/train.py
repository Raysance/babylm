import json
import os
import sys

import sentencepiece as spm
import torch
from datasets import DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, get_constant_schedule_with_warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hf_utils import save_model_and_tokenizer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DATA_PATH = os.path.join(WORKSPACE_ROOT, "train-set", "tokenized", "train-formatted.arrow")
TOKENIZER_PATH = os.path.join(WORKSPACE_ROOT, "tokenizer", "chinese_spm.model")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "models", "pretrain", "encoder")
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "logs", "encoder_pretrain")

MAX_LEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SAVE_STEPS = 25000
LOGGING_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
MLM_PROBABILITY = 0.15
WARMUP_STEPS = 30000
LOAD_OPTIMIZER_STATE = True
TRAINER_STATE_NAME = "trainer_state.json"
DATALOADER_NUM_WORKERS = 2

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
    for row_idx in range(masked_indices.size(0)):
        if masked_indices[row_idx].any():
            continue
        candidate_indices = (~special_tokens_mask[row_idx]).nonzero(as_tuple=False).flatten()
        if candidate_indices.numel() > 0:
            random_position = candidate_indices[torch.randint(candidate_indices.numel(), (1,)).item()]
            masked_indices[row_idx, random_position] = True

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


def load_saved_step(model_dir):
    state_path = os.path.join(model_dir, TRAINER_STATE_NAME)
    if not os.path.exists(state_path):
        return None

    with open(state_path, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    return int(state.get("global_step", 0))


def save_trainer_state(output_dir, global_step):
    state_path = os.path.join(output_dir, TRAINER_STATE_NAME)
    with open(state_path, "w", encoding="utf-8") as state_file:
        json.dump({"global_step": global_step}, state_file, ensure_ascii=False, indent=2)


def save_training_checkpoint(model, optimizer, scheduler, output_dir, global_step):
    save_model_and_tokenizer(model, TOKENIZER_PATH, output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    save_trainer_state(output_dir, global_step)


def choose_resume_source(final_dir, latest_checkpoint):
    final_saved_step = load_saved_step(final_dir) if os.path.exists(final_dir) else None
    if final_saved_step is not None:
        return final_dir, final_saved_step

    if latest_checkpoint is not None:
        checkpoint_step, checkpoint_dir = latest_checkpoint
        return checkpoint_dir, checkpoint_step

    return None, 0


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
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    final_dir = os.path.join(OUTPUT_DIR, "final")
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    resume_dir, start_step = choose_resume_source(final_dir, latest_checkpoint)
    if resume_dir == final_dir:
        if is_main_process:
            print(f"Continuing training from final model at step {start_step}: {final_dir}")
        model = BertForMaskedLM.from_pretrained(resume_dir).to(device)
    elif resume_dir is not None:
        if is_main_process:
            print(f"Resuming from checkpoint at step {start_step}: {resume_dir}")
            if os.path.exists(final_dir):
                print(f"Ignoring final without {TRAINER_STATE_NAME}: {final_dir}")
        model = BertForMaskedLM.from_pretrained(resume_dir).to(device)
    else:
        if is_main_process:
            print("No final model or checkpoint found. Starting encoder training from scratch.")
        model = get_encoder_model(vocab_size)
        model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
    )
    if resume_dir is not None and LOAD_OPTIMIZER_STATE:
        optimizer_path = os.path.join(resume_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE
        scheduler_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=True))

    model.train()
    global_step = start_step
    recent_losses = []
    progress_bar = tqdm(initial=start_step, disable=not is_main_process, unit="step")

    if is_main_process:
        print("Encoder will keep training until Ctrl+C.")
        print(f"Warmup steps: {WARMUP_STEPS}; scheduler: constant after warmup.")
        print(f"Periodic checkpoints will be saved every {SAVE_STEPS} steps.")
        print(f"Ctrl+C will overwrite final model: {final_dir}")

    try:
        while True:
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = model(**batch)
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

                display_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                recent_losses.append(display_loss)
                if len(recent_losses) > LOGGING_STEPS:
                    recent_losses.pop(0)

                loss.backward()

                if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if is_main_process:
                    progress_bar.update(1)
                    if global_step % LOGGING_STEPS == 0:
                        avg_loss = sum(recent_losses) / len(recent_losses)
                        progress_bar.set_postfix({
                            "loss": f"{display_loss:.4f}",
                            f"avg{len(recent_losses)}": f"{avg_loss:.4f}",
                        })

                if global_step % SAVE_STEPS == 0:
                    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    save_training_checkpoint(model, optimizer, scheduler, checkpoint_dir, global_step)
                    if is_main_process:
                        print(f"\nSaved checkpoint: {checkpoint_dir}")
    except KeyboardInterrupt:
        if is_main_process:
            print("\nCtrl+C received. Saving current progress to final...")
            save_training_checkpoint(model, optimizer, scheduler, final_dir, global_step)
            print(f"Final model overwritten: {final_dir}")
    finally:
        if is_main_process:
            progress_bar.close()


if __name__ == "__main__":
    main()
