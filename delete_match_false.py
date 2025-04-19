#!/usr/bin/env python3
import json
import os
import sys

def filter_match_false(input_file, output_file=None):
    """
    从JSON文件中过滤掉execution.match_status为false的数据元素
    :param input_file: 输入的JSON文件路径
    :param output_file: 输出的JSON文件路径，默认为None时覆盖原文件
    """
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return False
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = 0
        filtered_count = 0
        
        if isinstance(data, dict):
            # 如果是单个对象
            original_count = 1
            if data.get('execution', {}).get('match_status') == False:
                print(f"警告：整个JSON对象的match_status为false，不会输出结果")
                filtered_count = 0
                return False
            filtered_count = 1
        elif isinstance(data, list):
            # 如果是数组，过滤掉execution.match_status为false的元素
            original_count = len(data)
            filtered_data = []
            
            for item in data:
                # 检查嵌套的execution.match_status字段
                if item.get('execution', {}).get('match_status') == False:
                    continue  # 跳过match_status为false的元素
                filtered_data.append(item)
            
            filtered_count = len(filtered_data)
            data = filtered_data
        
        print(f"原始数据元素数量: {original_count}")
        print(f"过滤后的数据元素数量: {filtered_count}")
        print(f"删除的数据元素数量: {original_count - filtered_count}")
        
        # 确定输出文件路径
        if output_file is None:
            output_file = input_file
        
        # 写入结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"已成功将结果保存到 {output_file}")
        return True
    
    except json.JSONDecodeError:
        print(f"错误：{input_file} 不是有效的JSON文件")
    except Exception as e:
        print(f"错误：{str(e)}")
    
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python delete_match_false.py <输入文件> [输出文件]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not filter_match_false(input_file, output_file):
        sys.exit(1)
