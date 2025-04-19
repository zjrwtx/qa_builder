#!/usr/bin/env python3
import json
import os
import sys

def remove_execution_field(json_file_path):
    """
    删除JSON文件中所有数据元素的execution字段
    
    Args:
        json_file_path: JSON文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件 '{json_file_path}' 不存在")
        return False
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据结构是否为列表
        if isinstance(data, list):
            # 遍历列表中的每个元素并删除"execution"字段
            modified = False
            for item in data:
                if isinstance(item, dict) and 'execution' in item:
                    del item['execution']
                    modified = True
        # 检查数据结构是否为字典且包含列表
        elif isinstance(data, dict):
            # 查找字典中的列表并处理
            modified = False
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'execution' in item:
                            del item['execution']
                            modified = True
                elif isinstance(value, dict) and 'execution' in value:
                    del data[key]['execution']
                    modified = True
        else:
            print(f"警告: 文件 '{json_file_path}' 的数据结构不是预期的列表或字典")
            return False
        
        if not modified:
            print(f"警告: 文件 '{json_file_path}' 中没有找到execution字段")
            return False
        
        # 写回处理后的JSON数据
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"成功: 已删除文件 '{json_file_path}' 中的execution字段")
        return True
    
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_file_path}' 不是有效的JSON格式")
    except Exception as e:
        print(f"错误: 处理文件 '{json_file_path}' 时出现问题: {str(e)}")
    
    return False

def main():
    if len(sys.argv) < 2:
        print("用法: python delete_execution.py <json_file_path> [<json_file_path> ...]")
        return
    
    for file_path in sys.argv[1:]:
        remove_execution_field(file_path)

if __name__ == "__main__":
    main() 