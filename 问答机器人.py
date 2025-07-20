from sentence_transformers import SentenceTransformer, util
import torch
import json

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
        """高血压（Hypertension）是指动脉血压持续升高的一种慢性非传染性疾病。根据最新的诊断标准，
收缩压≥140 mmHg和/或舒张压≥90 mmHg，或正在服用降压药物的患者可诊断为高血压。
高血压是心血管疾病的重要风险因素，长期血压升高可导致心脏病、脑卒中、肾衰竭等器官损害。
高血压分为原发性（约占90%-95%，病因复杂且不明）和继发性（由肾脏疾病、内分泌疾病等引起）。
多数患者早期无明显临床表现，被称为“无声杀手”，部分患者可能出现头痛、头晕、心悸等症状。
治疗包括生活方式干预（低盐饮食、减重、戒烟、运动）、药物治疗（钙通道阻滞剂、ACE抑制剂、利尿剂等），
目标在于有效控制血压，预防心脑肾等并发症。患者需定期监测血压，遵医嘱用药，保持长期管理。"""
    ),
    # … 在这里继续添加其他问答对 …
}

# 2. 加载 Sentence-BERT 多语言模型
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)

# 3. 预先计算标准问题的 embeddings
questions = list(qa_dict.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)

def answer_question(user_question, top_k=1):
    """
    输入用户问题，返回 top_k 个匹配结果，结果为字典列表
    每个字典包含:
      - matched_question: 匹配到的标准问题
      - answer: 对应答案文本
      - cosine_score: 余弦相似度（保留 4 位小数）
    """
    # 编码用户问题
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    # 计算余弦相似度
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    # 取 top_k 最大
    top_k = min(top_k, len(questions))
    topk_scores, topk_idxs = torch.topk(cosine_scores, k=top_k, largest=True)

    results = []
    for score, idx in zip(topk_scores, topk_idxs):
        idx = idx.item()
        score = score.item()
        matched_q = questions[idx]
        matched_a = qa_dict[matched_q]
        results.append({
            "matched_question": matched_q,
            "answer": matched_a,
            "cosine_score": round(score, 4)
        })

    if not results:
        results.append({
            "matched_question": None,
            "answer": "抱歉，未能找到匹配的答案。请尝试换个问法。",
            "cosine_score": 0.0
        })
    return results

if __name__ == "__main__":
    print("欢迎使用多语言问答系统，输入 exit 或 Ctrl+C 退出。")
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

        answers = answer_question(query, top_k=1)
        # 以 JSON 形式打印
        print(json.dumps(answers, ensure_ascii=False, indent=2))
