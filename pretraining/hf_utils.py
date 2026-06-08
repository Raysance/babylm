import os

from transformers import LlamaTokenizer


def build_tokenizer(tokenizer_path):
    return LlamaTokenizer(
        vocab_file=tokenizer_path,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
    )


def save_model_and_tokenizer(model, tokenizer_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    build_tokenizer(tokenizer_path).save_pretrained(output_dir)
