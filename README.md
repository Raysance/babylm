This project is created for homework of SJTU (2025-2026-2)-CS3966-01(NLP and LLM)

This project is to participate in [CN BabyLM Challenge](https://chinese-babylm.github.io/).

[Hugging Face of Chinese BabyLM](https://huggingface.co/chinese-babylm-org)
[Eval pipeline of Chinese BabyLM](https://github.com/SiyuanSong2004/chinese-babylm-eval-pipeline)

## NLU Track pipeline

The project uses one shared SentencePiece tokenizer and two task-specific
pretraining backbones:

- Decoder (`pretraining/decoder/train.py`): causal LM for ZhoBLiMP minimal pairs.
- Encoder (`pretraining/encoder/train.py`): MLM for CLUE fine-tuning.

Run preprocessing once:

```powershell
E:\Miniconda3\envs\babylmm\python.exe .\preprocess\cache_dataset.py
```

Train the decoder:

```powershell
E:\Miniconda3\envs\babylmm\python.exe .\pretraining\decoder\train.py
```

Train the encoder:

```powershell
E:\Miniconda3\envs\babylmm\python.exe .\pretraining\encoder\train.py
```

Both training scripts save Hugging Face-compatible final models with tokenizer
files under:

- `models/pretrain/decoder/final`
- `models/pretrain/encoder/final`

Evaluate NLU:

```powershell
bash .\scripts\eval_nlu_decoder.sh
bash .\scripts\eval_nlu_encoder.sh
```

If a checkpoint was created before tokenizer export was added, attach tokenizer
files manually:

```powershell
E:\Miniconda3\envs\babylmm\python.exe .\scripts\export_tokenizer.py --model_dir .\models\pretrain\decoder\checkpoint-25000
```
