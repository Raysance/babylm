# 检查编码
import chardet

with open('../untokenized/train-formatted.txt', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(f"检测到的编码: {result}")

# 如果不是 UTF-8，转换
if result['encoding'].lower() != 'utf-8':
    text = raw_data.decode(result['encoding'])
    with open('../untokenized/train-formatted.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("已转换为 UTF-8")