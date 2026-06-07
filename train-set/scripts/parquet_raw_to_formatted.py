import pandas as pd
import re
import sys

def is_pinyin_dialogue(text):
    """
    识别拼音对话：通常包含 *NAME: 且内容主要是带数字的拼音或大量英文字母
    """
    if re.match(r'^\s*\*.*:', text):
        # 去除标签后检查剩余内容
        content = re.sub(r'^\s*\*.*:', '', text).strip()
        # 如果包含带音调数字的拼音(如 wei2)或基本全是英文/数字且不含中文
        if re.search(r'[a-z]+\d', content, re.I) or (not re.search(r'[\u4e00-\u9fa5]', content)):
            return True
    return False

def clean_latex(text):
    """
    处理 LaTeX：保留表达式中的文字内容，并移除内部及周围的空格
    """
    # 找到所有 $$...$$ 或 $...$
    def replace_math(match):
        # match.group(1) 是公式本身 ($...$ 或 $$...$$)
        formula = match.group(1)
        # 1. 移除 LaTeX 命令 (如 \left, \text, \frac 等)
        formula = re.sub(r'\\[a-zA-Z]+', '', formula)
        # 2. 核心要求：移除内部空格、换行
        formula = formula.replace(' ', '').replace('\t', '').replace('\n', '')
        # 3. 移除 LaTeX 常见的特殊修饰符号
        formula = re.sub(r'[\{\}\$\(\)\^\_\+\-\=\[\]\|\\]', '', formula)
        return formula
    
    # 修改正则：\s*(\$\$.*?\$\$|\$.*?\$)\s* 
    # 这样会捕获公式及其前后的单个或多个空格，并在替换时将其剔除，防止公式被空格切断
    cleaned = re.sub(r'\s*(\$\$.*?\$\$|\$.*?\$)\s*', replace_math, text, flags=re.DOTALL)
    return cleaned

def split_sentences(text):
    """
    切分句子：
    1. 按照 \n 和空格切分
    2. 按照常见中文标点切分
    """
    # 先处理 LaTeX 以防干扰
    text = clean_latex(text)
    # 将空格、换行以及常用标点统一替换为特殊分隔符
    # 标点符号：。！？；!?;
    text = re.sub(r'[\n\s。！？；!?;]', '|', text)
    return [s.strip() for s in text.split('|') if s.strip()]

def final_clean(sentence):
    """
    最终清洗：
    1. 移除对话前缀 *ADULT: 等
    2. 忽略每行开头的连续数字字母 (如 123A你好 -> 你好)
    3. 移除所有标点（包括书名号、引号等），仅保留汉字、数字、英文字母
    """
    # 1. 移除对话前缀
    sentence = re.sub(r'^\s*\*.*?:', '', sentence)
    
    # 2. 要求：忽略每行开头的连续数字字母
    sentence = re.sub(r'^[a-zA-Z0-9]+', '', sentence)
    
    # 3. 移除所有非 汉字、数字、字母 的内容（去除标点符号）
    # 包括 《》 "" '' （ ）等
    sentence = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', sentence)
    return sentence

def process_parquet(input_file, output_file, text_column='text'):
    print(f"开始处理: {input_file}")
    df = pd.read_parquet(input_file)
    
    all_cleaned_sentences = []
    
    for _, row in df.iterrows():
        raw_text = str(row[text_column])
        
        # 1. 过滤 Wiki 标签行 (如 |+ |- {| |! ====等)
        lines = raw_text.split('\n')
        filtered_lines = []
        for line in lines:
            line_s = line.strip()
            # 过滤 Wiki 表格/标题符号
            if re.match(r'^[\{\|\|\+\-\!]|^\s*={2,}', line_s):
                continue
            # 2. 忽略拼音对话
            if is_pinyin_dialogue(line_s):
                continue
            filtered_lines.append(line)
        
        textblock = " ".join(filtered_lines)
        
        # 3. 切分句子并逐句清洗
        sentences = split_sentences(textblock)
        for s in sentences:
            clean_s = final_clean(s)
            
            # 4. 过滤条件
            length = len(clean_s)
            if length > 3:
                # 要求：过滤排除数字字母占比高于 50% 的数据行
                alnum_chars = re.findall(r'[a-zA-Z0-9]', clean_s)
                alnum_ratio = len(alnum_chars) / length if length > 0 else 0
                
                if alnum_ratio <= 0.5:
                    all_cleaned_sentences.append(clean_s)
    
    # 5. 去重
    print(f"清洗完成，正在去重...")
    result_df = pd.DataFrame(all_cleaned_sentences, columns=['text'])
    result_df.drop_duplicates(inplace=True)
    
    print(f"最终获取数据条数: {len(result_df)}")
    result_df.to_parquet(output_file, index=False)
    print(f"已保存至: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python parquet_raw_to_formatted.py <输入文件> <输出文件> [列名]")
    else:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
        col = sys.argv[3] if len(sys.argv) > 3 else 'text'
        process_parquet(in_file, out_file, col)