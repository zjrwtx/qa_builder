import gradio as gr
import json
import os
from typing import List, Dict, Tuple, Optional

# 存储所有 QA 对的列表
qa_pairs: List[Dict[str, str]] = []

def add_qa(question: str, answer: str) -> str:
    """添加新的 QA 对并返回当前所有 QA 对的 JSON 字符串"""
    if question and answer:  # 确保问题和答案都不为空
        qa_pairs.append({
            "question": question,
            "final_answer": answer
        })
    return json.dumps(qa_pairs, ensure_ascii=False, indent=2)

def delete_qa(index: Optional[int]) -> str:
    """删除指定索引的 QA 对并返回当前所有 QA 对的 JSON 字符串"""
    global qa_pairs
    print(f"尝试删除索引 {index}, 当前有 {len(qa_pairs)} 个 QA 对")
    
    # 确保索引是有效的整数
    if index is None:
        print("删除失败：索引为 None")
        return json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        
    if 0 <= index < len(qa_pairs):
        del qa_pairs[index]
        print(f"删除成功，剩余 {len(qa_pairs)} 个 QA 对")
    else:
        print(f"删除失败：索引 {index} 超出范围")
    return json.dumps(qa_pairs, ensure_ascii=False, indent=2)

def get_qa_list():
    """返回 QA 对的列表，用于显示在下拉框中"""
    return [f"{i}: {qa['question'][:50]}..." for i, qa in enumerate(qa_pairs)]

def save_to_json():
    """将 QA 对保存到 JSON 文件"""
    with open("qa_data.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    return "数据已保存到 qa_data.json"

def import_from_json(file) -> Tuple[str, list]:
    """从 JSON 文件导入 QA 对"""
    global qa_pairs
    if file is None:
        return "未选择文件", []
    
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
            qa_pairs = loaded_data
        return f"成功从 {os.path.basename(file.name)} 导入 {len(qa_pairs)} 个 QA 对", get_qa_list()
    except Exception as e:
        return f"导入失败: {str(e)}", []

def load_and_display_qa() -> Tuple[str, list]:
    """加载并显示 QA 数据"""
    return json.dumps(qa_pairs, ensure_ascii=False, indent=2), get_qa_list()

def process_delete_selection(selection: str) -> Optional[int]:
    """从下拉框选择中提取索引值"""
    print(f"收到选择的值: '{selection}'")
    if selection is None or selection == "":
        print("无效选择：为空")
        return None
    
    try:
        # 从选择中提取索引部分
        index_str = selection.split(":")[0].strip()
        index = int(index_str)
        print(f"解析选择的索引: {index}")
        return index
    except (ValueError, IndexError, AttributeError) as e:
        print(f"解析索引出错: {str(e)}, 选择值: '{selection}'")
        return None

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# QA 数据收集工具")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="问题", placeholder="请输入问题")
            answer_input = gr.Textbox(label="答案", placeholder="请输入标准答案")
            add_btn = gr.Button("添加 QA 对")
        
        with gr.Column():
            json_output = gr.Textbox(label="当前所有 QA 对", lines=10)
            save_btn = gr.Button("保存为 JSON 文件")
            save_status = gr.Textbox(label="保存状态", interactive=False)
    
    with gr.Row():
        with gr.Column():
            qa_dropdown = gr.Dropdown(label="选择要删除的 QA 对", choices=[], interactive=True)
            delete_btn = gr.Button("删除选择的 QA 对")
            delete_status = gr.Textbox(label="删除状态", interactive=False)
            
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="导入 JSON 文件", file_types=[".json"])
            import_btn = gr.Button("导入数据")
            import_status = gr.Textbox(label="导入状态", interactive=False)
    
    # 绑定事件处理函数
    add_btn.click(
        fn=add_qa,
        inputs=[question_input, answer_input],
        outputs=json_output
    ).then(
        fn=get_qa_list,
        outputs=qa_dropdown
    )
    
    # 删除功能
    def update_delete_status(index):
        if index is None:
            return "删除失败：未选择有效的 QA 对"
        return f"成功删除索引为 {index} 的 QA 对"
    
    # 使用正确的方式处理删除操作
    def handle_delete(selection):
        index = process_delete_selection(selection)
        json_str = delete_qa(index)
        # 同时返回三个值：JSON输出、下拉框选项和状态信息
        return json_str, get_qa_list(), update_delete_status(index)
    
    delete_btn.click(
        fn=handle_delete,
        inputs=[qa_dropdown],
        outputs=[json_output, qa_dropdown, delete_status]
    )
    
    save_btn.click(
        fn=save_to_json,
        outputs=save_status
    )
    
    import_btn.click(
        fn=import_from_json,
        inputs=file_input,
        outputs=[import_status, qa_dropdown]
    ).then(
        fn=load_and_display_qa,
        outputs=[json_output, qa_dropdown]
    )

    # 在启动时加载数据到界面
    demo.load(
        fn=load_and_display_qa,
        inputs=None,
        outputs=[json_output, qa_dropdown]
    )

if __name__ == "__main__":

    
    demo.launch() 