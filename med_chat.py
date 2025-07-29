#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
安装依赖：
pip install torch faiss-cpu sentence-transformers numpy
使用领域微调模型和 HNSW 索引，保证检索准确度与实时性。
检索结果仅作辅助参考，实际诊疗请依临床规范和个人经验。

使用 FAISS HNSW 索引 + BioBERT-STSB 模型做语义检索。
"""

import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# —— 配置 ——  
MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
#这是医疗特定领域的模型，擅长医学领域。
INDEX_FILE = "med_index_hnsw.faiss"
EMB_FILE   = "med_emb.npy"
TXT_FILE   = "med_data.txt"
HNSW_M     = 32
HNSW_EFCON = 100
HNSW_EFSEA = 50
TOP_K      = 2  # 每次返回2条

# —— 预存“更准确”的医学知识清单（10 条） ——  
PREDEFINED_TEXTS = [
    "急性心肌梗死：持续性压榨样胸痛＞20分钟，可伴大汗、恶心、心悸和放射痛，需紧急经皮冠状动脉介入。",
    "肺炎的典型症状：发热、咳嗽（多为脓痰）、呼吸急促，体检可闻及湿啰音或支气管呼吸音。",
    "深静脉血栓形成：下肢单侧肿胀、压痛、皮温升高、表浅静脉怒张，需行下肢静脉超声检查。",
    "糖尿病酮症酸中毒：多饮、多尿、脱水、呼吸深快（库斯莫尔呼吸）、血糖＞16.7 mmol/L，血酮阳性。",
    "卒中（中风）急性期：忽然出现面、臂、腿偏瘫，言语不清，需40分钟内咨询卒中绿色通道并行头颅CT。",
    "慢性阻塞性肺疾病（COPD）：长期咳嗽、咳痰和/或呼吸困难，急性加重期可出现呼吸衰竭。",
    "心力衰竭常见症状：劳力性呼吸困难、端坐呼吸、夜间阵发性呼吸困难及下肢水肿。",
    "肾衰竭：早期乏力、恶心、食欲不振，晚期可出现高钾血症、代谢性酸中毒和尿毒症状。",
    "肝硬化可能并发门脉高压：食管胃底静脉曲张、腹水、脾大及肝性脑病。",
    "蜂窝织炎：皮肤红肿热痛，边界不清，多伴全身发热，需早期给予抗生素治疗。"
]

# —— 初始化模型 ——  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# —— 构建或加载索引 ——  
if not os.path.exists(INDEX_FILE):
    # 如果无文本文件，则写入预存文本
    if not os.path.exists(TXT_FILE):
        with open(TXT_FILE, "w", encoding="utf-8") as f:
            for line in PREDEFINED_TEXTS:
                f.write(line + "\n")

    # 读取文本
    texts = []
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)

    # 编码并归一化
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    # 建立 HNSW 索引
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EFCON
    index.hnsw.efSearch       = HNSW_EFSEA
    index.add(embeddings)

    # 持久化
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, embeddings)
else:
    # 加载索引、向量、文本
    index     = faiss.read_index(INDEX_FILE)
    embeddings= np.load(EMB_FILE)
    texts     = []
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)

# —— 核心检索函数 ——  
def retrieve(query, top_k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for i in range(top_k):
        idx = int(I[0][i])
        score = float(D[0][i])
        results.append((texts[idx], score))
    return results

# —— 持续对话主循环 ——  
def chat_loop():
    print("=== 医生辅助知识检索系统 ===")
    print("输入患者主诉或医学疑问，按 Enter。输入“exit”退出。")
    while True:
        query = input("\n医生 > ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("再见！")
            break
        hits = retrieve(query, TOP_K)
        print("\n系统推荐（Top %d）：" % TOP_K)
        for text, score in hits:
            print("  [{:.4f}] {}".format(score, text))

if __name__ == "__main__":
    chat_loop()
