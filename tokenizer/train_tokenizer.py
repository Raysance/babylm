import sentencepiece as spm

print("开始训练 SentencePiece 分词器，包含 [MASK] token...")
spm.SentencePieceTrainer.train(
    input='../../train-formatted.txt',
    model_prefix='chinese_spm',
    vocab_size=30000,
    character_coverage=0.9995,
    model_type='bpe',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[BOS]',
    eos_piece='[EOS]',
    num_threads=4,
    user_defined_symbols=['[MASK]'],   # 关键：添加 [MASK]
)

print("分词器训练完成！")
