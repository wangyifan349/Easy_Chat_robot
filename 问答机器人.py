import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

# 1. 构造问答对字典
qa_dict = {
    "如何计算余弦相似度？": (
        """余弦相似度计算公式如下：

def cosine_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b)

def l2_distance(vec1, vec2):
    # 计算两个向量的L2距离（欧氏距离）
    diff = vec1 - vec2
    return torch.norm(diff, p=2)

余弦相似度值域为[-1,1]，表示两个向量的方向相似程度。"""
    ),
    "高血压是什么？": (
        """高血压（Hypertension）是指动脉血压持续升高的一种慢性非传染性疾病…"""
    ),
    # … 在这里继续添加其它问答对 …
}

# 2. 加载 Sentence-BERT 多语言模型
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)

# 3. 准备标准问题向量，并做 L2 归一化
questions = list(qa_dict.keys())
question_embs = model.encode(questions, convert_to_tensor=False, normalize_embeddings=True)  # normalize_embeddings=True 会自动 L2 归一化
question_embs = np.array(question_embs).astype('float32')

# 4. 用 FAISS 建立索引 (IndexFlatIP 用内积做相似度检索)
d = question_embs.shape[1]    # 向量维度
index = faiss.IndexFlatIP(d)
index.add(question_embs)      # 批量添加所有标准问题向量

def answer_question_faiss(user_question, top_k=1):
    """
    使用 FAISS 检索 top_k 答案
    返回格式：list of dict, 每 dict 包含 matched_question, answer, cosine_score
    """
    # 1）编码并归一化用户问题向量
    user_emb = model.encode([user_question], convert_to_tensor=False, normalize_embeddings=True)
    user_emb = np.array(user_emb).astype('float32')

    # 2）FAISS 检索
    scores, idxs = index.search(user_emb, top_k)  # scores 是内积，相当于余弦相似度
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    results = []
    for score, idx in zip(scores, idxs):
        matched_q = questions[idx]
        matched_a = qa_dict[matched_q]
        results.append({
            "matched_question": matched_q,
            "answer": matched_a,
            "cosine_score": round(float(score), 4)
        })

    if not results:
        results.append({
            "matched_question": None,
            "answer": "抱歉，未能找到匹配的答案。请尝试换个问法。",
            "cosine_score": 0.0
        })
    return results

if __name__ == "__main__":
    print("欢迎使用基于 FAISS 的多语言问答系统，输入 exit 或 Ctrl+C 退出。")
    while True:
        try:
            query = input("请输入你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出问答系统。")
            break

        if not query:
            print("输入不能为空，请重新输入。")
            continue
        if query.lower() == "exit":
            print("退出问答系统。")
            break

        answers = answer_question_faiss(query, top_k=1)
        # JSON 格式化输出
        print(json.dumps(answers, ensure_ascii=False, indent=2))
