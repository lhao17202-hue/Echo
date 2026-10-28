# -*- coding: utf-8 -*-
"""
hello.py
功能：
1. 在控制台打印 "hello"
2. 将 "hello" 写入桌面文件 C:\\Users\\VK\\Desktop\\hello.txt
"""

import os

# 目标文件路径（Windows 桌面）
DESKTOP_FILE = r"C:\Users\VK\Desktop\hello.txt"

def main():
    # 1. 在控制台打印 hello
    print("hello")

    # 2. 写入内容到桌面文件
    # 确保目标目录存在（若桌面目录缺失则自动创建）
    target_dir = os.path.dirname(DESKTOP_FILE)
    os.makedirs(target_dir, exist_ok=True)

    # 以 UTF-8 编码写入文件，避免中文系统下编码问题
    with open(DESKTOP_FILE, "w", encoding="utf-8") as f:
        f.write("hello")

    print(f"已写入文件: {DESKTOP_FILE}")

if __name__ == "__main__":
    main()
