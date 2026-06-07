import pandas as pd
import sys
import os

def parquet_to_txt(input_file, output_file, col_name='sentence'):
    """
    将 Parquet 文件的指定列导出为纯文本文件，每行一条数据。
    """
    try:
        print(f"读取 Parquet 文件: {input_file} ...")
        df = pd.read_parquet(input_file)
        
        if col_name not in df.columns:
            print(f"错误: 列名 '{col_name}' 不存在。")
            print(f"可选列名为: {df.columns.tolist()}")
            return

        print(f"导出列 '{col_name}' 到 {output_file} ...")
        
        # 将该列内容转换为字符串并逐行写入
        # 使用 utf-8 编码以支持中文
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in df[col_name]:
                # 确保内容为字符串且不含换行符干扰（虽然清洗脚本应该已经处理过）
                line = str(item).replace('\n', ' ').strip()
                if line:
                    f.write(line + '\n')
        
        print(f"导出成功！共写入 {len(df)} 行。")

    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python parquet_to_txt.py <输入Parquet> <输出Txt> [列名]")
        print("示例: python parquet_to_txt.py cleaned_data.parquet output.txt sentence")
    else:
        in_p = sys.argv[1]
        out_t = sys.argv[2]
        col = sys.argv[3] if len(sys.argv) > 3 else 'sentence'
        parquet_to_txt(in_p, out_t, col)
