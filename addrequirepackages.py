import json

def add_require_dependencies():
    # 读取JSON文件
    try:
        with open('qa_data_with_code.json', 'r') as file:
            data = json.load(file)
            
        # 在每个项目的metadata中添加require_dependencies字段
        for item in data:
            if 'metadata' in item:
                item['metadata']['require_dependencies'] = ['medcalc-bench']
            else:
                pass
                
        # 将更新后的数据写回文件
        with open('qa_data_with_code.json', 'w') as file:
            json.dump(data, file, indent=2)
            
        print(f"成功更新 {len(data)} 个项目的metadata字段")
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    add_require_dependencies()
