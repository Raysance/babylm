#!/usr/bin/env python3
"""
Parquet 转 HuggingFace Arrow 格式
用法: python parquet_to_arrow.py --input ./data.parquet --output ./arrow_dataset
"""

import argparse
from datasets import load_dataset

def convert_parquet_to_arrow(input_path, output_path):
    """
    将 Parquet 文件或目录转换为 HuggingFace Arrow 格式
    
    Args:
        input_path: Parquet 文件路径或包含 parquet 文件的目录
        output_path: 输出的 Arrow 数据集目录
    """
    # 加载 parquet 文件
    print(f"正在加载 Parquet: {input_path}")
    dataset = load_dataset("parquet", data_files=input_path)
    
    # 保存为 Arrow 格式（自动生成 dataset_info.json, state.json 等）
    print(f"正在保存为 Arrow 格式: {output_path}")
    dataset.save_to_disk(output_path)
    
    print(f"✅ 转换完成！")
    print(f"   输出目录: {output_path}")
    print(f"   数据分片: {list(dataset.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parquet 转 HuggingFace Arrow 格式")
    parser.add_argument("--input", "-i", required=True, help="输入的 Parquet 文件或目录路径")
    parser.add_argument("--output", "-o", required=True, help="输出的 Arrow 数据集目录")
    
    args = parser.parse_args()
    convert_parquet_to_arrow(args.input, args.output)