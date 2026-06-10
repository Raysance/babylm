import json
import os

import sentencepiece as spm
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, get_constant_schedule_with_warmup

from hf_utils import save_model_and_tokenizer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DATA_PATH = os.path.join(WORKSPACE_ROOT, "train-set", "tokenized", "train-formatted.arrow")
TOKENIZER_PATH = os.path.join(WORKSPACE_ROOT, "tokenizer", "chinese_spm.model")

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


def make_masked_lm_inputs(input_ids):
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
    masked_input_ids = input_ids.clone()
    replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    masked_input_ids[replace_with_mask] = MASK_TOKEN_ID
    replace_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~replace_with_mask
    )
    random_words = torch.randint(MASK_TOKEN_ID + 1, VOCAB_SIZE, labels.shape, dtype=torch.long)
    masked_input_ids[replace_with_random] = random_words[replace_with_random]
    return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "labels": labels}


def variant_collate(batch):
    input_ids = [torch.tensor(item["input_ids"][:MAX_LEN], dtype=torch.long) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    mlm_batch = make_masked_lm_inputs(input_ids)
    causal_labels = input_ids[:, 1:].clone()
    causal_labels[causal_labels.eq(PAD_TOKEN_ID)] = -100
    return {
        **mlm_batch,
        "causal_input_ids": input_ids[:, :-1],
        "causal_attention_mask": input_ids[:, :-1].ne(PAD_TOKEN_ID).long(),
        "causal_labels": causal_labels,
    }


def causal_attention_mask(attention_mask):
    seq_len = attention_mask.size(1)
    lower_triangle = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long, device=attention_mask.device))
    return lower_triangle.unsqueeze(0) * attention_mask.unsqueeze(1)


def mlm_loss(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss


def normalized_elc_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states[1:]
    labels = batch["labels"]
    masked_positions = labels.ne(-100)
    target_ids = labels[masked_positions]
    if target_ids.numel() == 0:
        return outputs.last_hidden_state.sum() * 0.0
    total_loss = None
    for hidden_state in hidden_states:
        masked_hidden = hidden_state[masked_positions]
        prediction_scores = model.cls(masked_hidden)
        layer_loss = F.cross_entropy(prediction_scores, target_ids)
        total_loss = layer_loss if total_loss is None else total_loss + layer_loss
        del prediction_scores, masked_hidden
    return total_loss / len(hidden_states)


def normalized_gpt_bert_loss(model, batch):
    bidirectional_loss = mlm_loss(model, batch)
    causal_outputs = model(
        input_ids=batch["causal_input_ids"],
        attention_mask=causal_attention_mask(batch["causal_attention_mask"]),
        return_dict=True,
    )
    causal_logits = causal_outputs.logits
    causal_loss = F.cross_entropy(
        causal_logits.reshape(-1, causal_logits.size(-1)),
        batch["causal_labels"].reshape(-1),
        ignore_index=-100,
    )
    return 0.5 * (bidirectional_loss + causal_loss)


def mntp_loss(model, batch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True)
    labels = batch["labels"]
    masked_positions = labels.ne(-100)
    masked_positions[:, 0] = False
    target_ids = labels[masked_positions]
    if target_ids.numel() == 0:
        return outputs.logits.sum() * 0.0
    previous_positions = torch.zeros_like(masked_positions)
    previous_positions[:, :-1] = masked_positions[:, 1:]
    prediction_scores = outputs.logits[previous_positions]
    return F.cross_entropy(prediction_scores, target_ids)


def contrastive_weight_tying_loss(model, batch, temperature=0.05):
    outputs = model.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True)
    masked_positions = batch["labels"].ne(-100)
    target_ids = batch["labels"][masked_positions]
    if target_ids.numel() == 0:
        return outputs.last_hidden_state.sum() * 0.0
    hidden_states = outputs.last_hidden_state[masked_positions]
    target_embeddings = model.get_input_embeddings()(target_ids)
    hidden_states = F.normalize(hidden_states, dim=-1)
    target_embeddings = F.normalize(target_embeddings, dim=-1)
    logits = hidden_states @ target_embeddings.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits / temperature, labels)


def sync_mlm_head_to_embeddings(model):
    embeddings = model.get_input_embeddings().weight
    decoder = model.cls.predictions.decoder
    if decoder.weight.shape == embeddings.shape:
        decoder.weight = embeddings


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


def save_training_checkpoint(model, optimizer, scheduler, output_dir, global_step, variant, loss=None, avg_loss=None):
    save_model_and_tokenizer(model, TOKENIZER_PATH, output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    state = {"global_step": int(global_step), "variant": variant}
    if loss is not None:
        state["loss"] = float(loss)
    if avg_loss is not None:
        state["avg_loss"] = float(avg_loss)
    with open(os.path.join(output_dir, TRAINER_STATE_NAME), "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)


def choose_resume_source(final_dir, latest_checkpoint):
    final_saved_step = load_saved_step(final_dir) if os.path.exists(final_dir) else None
    if final_saved_step is not None:
        return final_dir, final_saved_step
    if latest_checkpoint is not None:
        checkpoint_step, checkpoint_dir = latest_checkpoint
        return checkpoint_dir, checkpoint_step
    return None, 0


def get_loss_fn(variant):
    if variant == "encoder-mlm":
        return mlm_loss
    if variant == "elc-bert":
        return normalized_elc_loss
    if variant == "gpt-bert":
        return normalized_gpt_bert_loss
    if variant == "cwt-bert":
        return contrastive_weight_tying_loss
    if variant == "mntp-bert":
        return mntp_loss
    raise ValueError(f"Unknown encoder objective variant: {variant}")


def train_variant(variant):
    output_dir = os.environ.get("BABYLM_MODEL_OUTPUT_DIR", os.path.join(WORKSPACE_ROOT, "models", variant))
    logs_dir = os.environ.get("BABYLM_LOGS_DIR", os.path.join(WORKSPACE_ROOT, "logs", f"{variant}_pretrain"))
    loss_fn = get_loss_fn(variant)
    is_main_process = int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]
    if is_main_process:
        print(f"Encoder {variant} training start. GPU count: {torch.cuda.device_count()}")
        print(f"Loading tokenizer: {TOKENIZER_PATH}")

    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    global VOCAB_SIZE
    VOCAB_SIZE = sp.GetPieceSize()

    dataset_or_dict = load_from_disk(TRAIN_DATA_PATH, keep_in_memory=False)
    train_dataset = dataset_or_dict["train"] if isinstance(dataset_or_dict, DatasetDict) else dataset_or_dict
    if is_main_process:
        print(f"Dataset rows: {len(train_dataset)}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_batch_size = 16 if variant == "elc-bert" else BATCH_SIZE
    effective_batch_size = int(os.environ.get("BABYLM_BATCH_SIZE", default_batch_size))
    if is_main_process:
        print(f"Batch size: {effective_batch_size}")
    dataloader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=variant_collate,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    final_dir = os.path.join(output_dir, "final")
    resume_dir, start_step = choose_resume_source(final_dir, find_latest_checkpoint(output_dir))
    if resume_dir is not None:
        if is_main_process:
            print(f"Resuming {variant} from step {start_step}: {resume_dir}")
        model = BertForMaskedLM.from_pretrained(resume_dir).to(device)
    else:
        if is_main_process:
            print(f"No {variant} checkpoint found. Starting from scratch.")
        model = get_encoder_model(VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)
    if resume_dir is not None and LOAD_OPTIMIZER_STATE:
        optimizer_path = os.path.join(resume_dir, "optimizer.pt")
        scheduler_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=True))

    model.train()
    global_step = start_step
    recent_losses = []
    progress_bar = tqdm(initial=start_step, disable=not is_main_process, unit="step")
    if is_main_process:
        print(f"{variant} will keep training until Ctrl+C.")
        print(f"Periodic checkpoints will be saved every {SAVE_STEPS} steps.")
        print(f"Ctrl+C will overwrite final model: {final_dir}")

    try:
        while True:
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    loss = loss_fn(model, batch) / GRADIENT_ACCUMULATION_STEPS
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
                        progress_bar.set_postfix({"loss": f"{display_loss:.4f}", f"avg{len(recent_losses)}": f"{avg_loss:.4f}"})
                if global_step % SAVE_STEPS == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    if variant == "cwt-bert":
                        sync_mlm_head_to_embeddings(model)
                    avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else display_loss
                    save_training_checkpoint(model, optimizer, scheduler, checkpoint_dir, global_step, variant, loss=display_loss, avg_loss=avg_loss)
                    if is_main_process:
                        print(f"\nSaved checkpoint: {checkpoint_dir}")
    except (KeyboardInterrupt, RuntimeError) as error:
        is_keyboard = isinstance(error, KeyboardInterrupt)
        is_oom = isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        if is_main_process and (is_keyboard or is_oom):
            print("\nSaving current progress to final...")
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if variant == "cwt-bert":
                sync_mlm_head_to_embeddings(model)
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
            last_loss = recent_losses[-1] if recent_losses else None
            save_training_checkpoint(model, optimizer, scheduler, final_dir, global_step, variant, loss=last_loss, avg_loss=avg_loss)
            print(f"Final model overwritten: {final_dir}")
        if not is_keyboard:
            raise
    finally:
        if is_main_process:
            progress_bar.close()
