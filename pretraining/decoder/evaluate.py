import os
import sys
import torch
import sentencepiece as spm
from transformers import GPT2LMHeadModel

# 将项目根目录添加到路径，确保可以导入 pretraining.decoder
# 当前文件在 babylm/pretraining/decoder/，根目录是向上两级
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==================== 1. 参数与路径配置 (软编码) ====================
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer/chinese_spm.model")
# 默认加载训练脚本输出的 final 模型
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/pretrain/decoder/final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN=128

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


def calculate_confidence(model, sp, text, device):
    """
    计算句子的置信度（整句话的对数概率之和）。
    """
    model.eval()
    # 按照 decoder.py 假设添加 BOS=2, EOS=3
    input_ids = [2] + sp.EncodeAsIds(text) + [3]
    input_tensor = torch.tensor([input_ids]).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor, labels=input_tensor)
        # 句子长度（含博 BOS/EOS）
        num_tokens = input_tensor.size(1)
        # Log-Likelihood Sum = -Loss * Tokens
        log_prob_sum = -outputs.loss.item() * num_tokens
        
    return log_prob_sum

def main():
    print(f"--- Decoder 模型评估工具 (设备: {DEVICE}) ---")

    # 1. 加载分词器
    if not os.path.exists(TOKENIZER_PATH):
        print(f"错误: 找不到分词器文件 {TOKENIZER_PATH}")
        return
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)

    # 2. 加载模型
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"错误: 在 {MODEL_PATH} 未找到已训练的模型。")
        print("请确认已经运行过 train.py 并成功生成了 final 模型。")
        return

    print(f"正在加载模型: {MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("模型加载成功！")

    # 3. 循环交互
    print("\n请输入句子进行评估（输入 'exit' 或 'quit' 退出）")
    while True:
        try:
            user_input = input("\n[输入句子] > ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("程序退出。")
                break
            
            confidence = calculate_confidence(model, sp, user_input, DEVICE)
            print(f"句子内容: {user_input}")
            print(f"置信度 (Log-Sum): {confidence:.4f}")
            
        except KeyboardInterrupt:
            print("\n检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"计算出错: {e}")

if __name__ == "__main__":
    main()
