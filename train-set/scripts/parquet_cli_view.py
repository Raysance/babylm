import pandas as pd
import sys

def preview(file_path, rows_per_page=10, col_name=None):
    try:
        print(f"正在加载 {file_path} ...")
        # 建议安装 pyarrow: pip install pyarrow
        df = pd.read_parquet(file_path)
        
        print(f"\n--- 文件信息: {file_path} ---")
        print(f"总行数: {len(df)}")
        print(f"所有列名: {df.columns.tolist()}")
        
        # 自动识别文本列
        if col_name is None:
            common_cols = ['text', 'content', 'body', 'sentence', 'raw']
            for c in common_cols:
                if c in df.columns:
                    col_name = c
                    break
            if col_name is None:
                col_name = df.columns[0]
        
        print(f"正在预览文本列: '{col_name}'")
        print("=" * 50)
        print("操作指令: [回车] 下一页 | [数字] 跳转到指定行 | [q] 退出")
        print("=" * 50)

        current_idx = 0
        while current_idx < len(df):
            end_idx = min(current_idx + rows_per_page, len(df))
            
            # 打印当前页内容
            for i in range(current_idx, end_idx):
                content = df.iloc[i][col_name]
                print(f"\nID: {i}")
                print(str(content).strip())
                print("-" * 30)

            # 交互提示
            user_input = input(f"\n已显示至 {end_idx}/{len(df)}。请输入指令: ").strip().lower()
            
            if user_input == 'q':
                print("程序已退出。")
                break
            elif user_input.isdigit():
                current_idx = int(user_input)
                print(f"--- 跳转至第 {current_idx} 行 ---")
            else:
                current_idx = end_idx
                if current_idx >= len(df):
                    print("已到达文件末尾。")

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    # 默认配置
    target_file = '../untokenized/train-formatted.parquet'
    page_size = 5
    
    # 命令行参数解析
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            page_size = int(sys.argv[2])
        except ValueError:
            print("警告: 行数参数必须是数字，已恢复默认值 5")
        
    preview(target_file, page_size)