import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "pretraining"))

from hf_utils import build_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--tokenizer_path",
        default=os.path.join(PROJECT_ROOT, "tokenizer", "chinese_spm.model"),
    )
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer_path)
    tokenizer.save_pretrained(args.model_dir)
    print(f"Tokenizer exported to {args.model_dir}")


if __name__ == "__main__":
    main()
