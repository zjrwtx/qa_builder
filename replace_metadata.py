#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os

def get_metadata():
    """返回要替换的元数据内容"""
    return {
        "license": "CC-BY-SA-4.0",
        "source": "https://huggingface.co/datasets/ncbi/MedCalc-Bench-v1.0",
        "domain": "medicine",
        "required_dependencies": [
            "medcalc-bench==1.7"
        ],
        "name": "Loong_medicine",
        "contributor": "Yifeng Wang ,Zikai Xiao,Fangyijie Wang",
        "date_created": "2025-04-19"
    }

def replace_metadata(target_file):
    """替换目标文件中的metadata字段"""
    try:
        # 读取目标文件
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取元数据
        metadata = get_metadata()
        
        # 如果数据是数组形式，则迭代每一项替换metadata字段
        if isinstance(data, list):
            for item in data:
                if 'metadata' in item:
                    item['metadata'] = metadata
        # 如果数据是对象形式，直接替换metadata字段
        elif isinstance(data, dict):
            if 'metadata' in data:
                data['metadata'] = metadata
        else:
            print(f"警告：在目标文件中没有找到metadata字段")
            return False
        
        # 写回文件
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return True
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python replace_metadata.py <target_json_file>")
        sys.exit(1)
    
    target_file = sys.argv[1]
    
    if not os.path.exists(target_file):
        print(f"目标文件不存在: {target_file}")
        sys.exit(1)
    
    # 替换目标文件中的元数据
    if replace_metadata(target_file):
        print(f"成功将元数据替换到文件: {target_file}")
    else:
        print(f"替换元数据失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
