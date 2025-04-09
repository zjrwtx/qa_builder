import asyncio
import json

from camel.verifiers import PythonVerifier

# 加载JSON数据
with open("qa_data_with_code.json", "r") as f:
    qa_data = json.load(f)

verifier = PythonVerifier(required_packages=["biopython"])
asyncio.run(verifier.setup(uv=True))


for i, qa_item in enumerate(qa_data):
    try:
        rationale = qa_item["rationale"]
        final_answer = qa_item["final_answer"]
        
        print(f"\n验证问题 {i+1}:")
        result = asyncio.run(
            verifier.verify(solution=rationale, reference_answer=final_answer)
        )
        print(f"结果: {result}")
        
    except Exception as e:
        print(f"问题 {i+1} 验证出错: {str(e)}")
        



# 清理资源
asyncio.run(verifier.cleanup())
