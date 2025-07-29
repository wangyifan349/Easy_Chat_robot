import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'   # 你想存的模型名或本地路径
MODEL_LOCAL_PATH = './saved_model'                     # 模型保存路径
EMBS_PATH = './questions_embs.npy'                      # 预计算向量保存路径
INDEX_PATH = './faiss.index'                            # faiss索引路径

# 1. 问答对字典
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
    # … 可以继续
}

questions = list(qa_dict.keys())

# 2. 模型加载或从本地加载
if os.path.exists(MODEL_LOCAL_PATH):
    print(f"从本地加载模型: {MODEL_LOCAL_PATH}")
    model = SentenceTransformer(MODEL_LOCAL_PATH)
else:
    print(f"下载模型: {MODEL_NAME}，并保存在: {MODEL_LOCAL_PATH}")
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_LOCAL_PATH)

# 3. 编码与索引加载 / 创建
if os.path.exists(EMBS_PATH) and os.path.exists(INDEX_PATH):
    # 加载预先计算向量和索引
    print("加载本地预训练向量和faiss索引...")
    question_embs = np.load(EMBS_PATH)
    index = faiss.read_index(INDEX_PATH)
else:
    # 重新计算向量并建索引，保存到本地
    print("计算问题向量并建立faiss索引...")
    question_embs = model.encode(questions, convert_to_tensor=False, normalize_embeddings=True)
    question_embs = np.array(question_embs).astype('float32')

    d = question_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(question_embs)

    # 保存
    np.save(EMBS_PATH, question_embs)
    faiss.write_index(index, INDEX_PATH)
    print("向量和索引保存完成。")

def answer_question_faiss(user_question, top_k=1):
    user_emb = model.encode([user_question], convert_to_tensor=False, normalize_embeddings=True)
    user_emb = np.array(user_emb).astype('float32')

    scores, idxs = index.search(user_emb, top_k)
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
        print(json.dumps(answers, ensure_ascii=False, indent=2))
