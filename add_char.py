import re
import sys
import os

def filter_and_add_chars(input_str, charset_file="charset.txt"):
    """
    处理输入字符串：过滤标点、拆分汉字，添加新字到字库文件
    :param input_str: 输入字符串
    :param charset_file: 字库文件名
    """
    # 1. 过滤标点符号和非汉字字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', input_str)
    
    if not chinese_chars:
        print("未检测到有效汉字")
        return
    
    # 2. 读取现有字库内容
    existing_chars = ""
    if os.path.exists(charset_file):
        with open(charset_file, 'r', encoding='utf-8') as f:
            existing_chars = f.read().strip()
    
    # 3. 识别新汉字
    new_chars = []
    seen_chars = set(existing_chars)  # 用于快速查找的集合
    
    for char in chinese_chars:
        if char not in seen_chars:
            new_chars.append(char)
            seen_chars.add(char)  # 避免重复添加
    
    # 4. 添加新字到文件
    if new_chars:
        # 去重并排序（可选）
        unique_new_chars = sorted(set(new_chars), key=lambda x: new_chars.index(x))
        
        with open(charset_file, 'a', encoding='utf-8') as f:
            f.write(''.join(unique_new_chars))
        
        print(f"添加了 {len(unique_new_chars)} 个新字: {''.join(unique_new_chars)}")
        print(f"字库更新完成，当前总字数: {len(existing_chars) + len(unique_new_chars)}")
    else:
        print("没有检测到新汉字")

if __name__ == "__main__":
    # 获取输入字符串
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    else:
        user_input = input("请输入要处理的文本: ")
    
    filter_and_add_chars(user_input)