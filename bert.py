from sentence_transformers import SentenceTransformer, util
import torch

from sentence_transformers import SentenceTransformer

# 1. 载入多语言Sentence-BERT模型
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)
# 2. 定义QA问答库，包括含代码块的示例，用原始字符串保持格式
qa_dict = {
    "如何计算余弦相似度？": (
        "余弦相似度计算公式如下：\n"
        r"""```python
def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b)
```
度量两个向量间的方向相似性，范围[-1,1]。"""
    ),
    "如何计算两数之和？": (
        "这里是一个简单的Python程序来计算两数之和：\n"
        r"""```python
def add_numbers(a, b):
    return a + b
```
调用`add_numbers(1, 2)`将返回3。"""
    ),
    # 更多问答对...
}

questions = list(qa_dict.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)

def answer_question(user_question: str, top_k=3) -> str:
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]

    # 获取相似度最高的top_k个问题
    top_k_scores, top_k_indices = torch.topk(cosine_scores, k=top_k, largest=True)

    # 收集前k个匹配的问题和答案
    results = []
    for i in range(top_k):
        idx = top_k_indices[i].item()
        score = top_k_scores[i].item()
        matched_question = questions[idx]
        matched_answer = qa_dict[matched_question]
        results.append(f"匹配问题: {matched_question}\n答案:\n{matched_answer}\n余弦相似度: {score:.4f}")
    
    return "\n\n".join(results) if results else "抱歉，未能找到匹配的答案。请尝试换个问法。"

if __name__ == "__main__":
    print("欢迎使用多语言问答系统，输入exit退出。")
    while True:
        query = input("请输入你的问题：")
        if query.strip().lower() == "exit":
            print("退出问答系统。")
            break
        response = answer_question(query, top_k=3)
        print(response)
