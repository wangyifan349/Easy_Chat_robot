# TF–IDF 与 BERT 精确对比

本文从数学定义、模型结构、特征维度、计算效率、可解释性和典型应用等方面，对经典文本向量化方法 TF–IDF 与预训练语言模型 BERT 进行全面对比，并给出基于 Hugging Face 社区模型的完整示例代码，帮助读者准确理解两者异同及使用场景。

---

## 一、数学定义

1. TF–IDF  
   - 设语料库包含 N 篇文档，词项 t 在语料库中出现的文档数为 df(t)。  
   - 逆文档频率：  
     IDF(t) = log (N / (1 + df(t)))  
   - 词频 TF(t, d) 为词 t 在文档 d 中的出现次数（或其归一化版本）。  
   - TF–IDF 权重：  
     w(t, d) = TF(t, d) × IDF(t)  
   - 文档表示为稀疏向量，维度等于词汇表大小。

2. BERT  
   - 基于 Transformer 编码器，输入 token 嵌入 + 可学习的位置编码。  
   - 多头自注意力 (Multi-Head Self-Attention) 捕捉全局上下文依赖。  
   - 预训练任务：  
     - 掩码语言模型 (Masked Language Modeling, MLM)  
     - 下一句预测 (Next Sentence Prediction, NSP)  
   - 输出为上下文相关的稠密向量（默认维度 768 或 1024）。

---

## 二、模型结构与特征维度

| 特性         | TF–IDF                       | BERT                                  |
|------------|-----------------------------|---------------------------------------|
| 表示类型      | 稀疏实数向量                    | 稠密实数向量                              |
| 维度         | 词汇表大小 (数万–数十万)           | 768 / 1024                            |
| 词序/上下文    | 不保留                          | 双向上下文敏感                            |
| 可解释性      | 高                             | 较低                                  |

---

## 三、计算效率与资源消耗

- TF–IDF  
  - 计算开销：O(#文档 × 平均文档长度)  
  - 内存占用：与词汇表和文档数线性相关  
  - 通常能在 CPU 上实时完成向量化与相似度检索

- BERT  
  - 预训练阶段：海量语料 + 多 GPU 多天计算  
  - 推理阶段（单文档编码）：O(L² × d)（L 为序列长度，d 为隐藏层维度）  
  - 需 GPU 加速，否则批量推理延迟较高

---

## 四、可解释性

- TF–IDF：权重计算透明，易于定位“关键”词汇及其影响  
- BERT：多层注意力与非线性映射，不易直观解释

---

## 五、典型应用场景

- TF–IDF  
  - 文本检索/相似度计算  
  - 基于线性模型的分类、聚类与特征工程  
  - 资源受限场景

- BERT  
  - 复杂文本分类（情感分析、意图识别）  
  - 序列标注（命名实体识别、分词）  
  - 问答系统、文本生成  
  - 需要深层语义理解与长距离依赖场景

---

## 六、代码示例

### 6.1 TF–IDF 实战

使用 scikit-learn 构建 TF–IDF 矩阵，并计算查询与文档相似度。

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载文本
docs = [
    "机器学习可以从数据中自动提取模型。",
    "深度学习是机器学习的一个分支，使用多层神经网络。",
    "TF IDF 是一种基于词频和逆文档频率的文本表示方法。"
]

# 2. 构建 TF–IDF 向量
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    norm='l2'
)
tfidf_matrix = vectorizer.fit_transform(docs)

# 3. 查询文本向量化 & 相似度计算
query = "深度神经网络模型"
query_vec = vectorizer.transform([query])
scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

# 4. 输出结果
rank = np.argsort(-scores)
print(f"查询句子：{query}\n")
for idx in rank:
    print(f"文档 {idx} | 相似度: {scores[idx]:.4f} | 内容: {docs[idx]}")
```

### 6.2 BERT 向量化示例

使用 Hugging Face 微调模型 `nlptown/bert-base-multilingual-uncased-sentiment`，提取句向量。

```python
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# 1. 准备 QA 知识库
qa_list = [
    {"question": "如何使用 Python 实现二分查找？",
     "answer": "在有序数组上定义左右指针，循环折半查找直到找到目标或区间为空。"},
    {"question": "什么是 TF–IDF？",
     "answer": "TF–IDF 是一种基于词频和逆文档频率的文本向量化方法，用于信息检索和特征提取。"},
    {"question": "怎样加载 Hugging Face 的预训练模型？",
     "answer": "使用 `AutoTokenizer.from_pretrained` 和 `AutoModel.from_pretrained` 即可加载分词器和模型。"},
    {"question": "如何在 PyTorch 中禁用梯度计算？",
     "answer": "在推理阶段使用 `with torch.no_grad():` 上下文管理器来禁用梯度。"},
]

questions = [item["question"] for item in qa_list]

# 2. 加载 BERT 分词器与模型
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# 3. 预计算知识库问题的向量（[CLS] 池化）
with torch.no_grad():
    encoded = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**encoded)
    qa_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

# 4. 定义检索函数
def retrieve_answer(user_question, top_k=1):
    enc = tokenizer([user_question], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    q_vec = out.last_hidden_state[:, 0, :].cpu().numpy()
    sims = cosine_similarity(q_vec, qa_vecs).flatten()
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for idx in idxs:
        results.append({
            "matched_question": qa_list[idx]["question"],
            "answer": qa_list[idx]["answer"],
            "score": float(sims[idx])
        })
    return results

# 5. 示例调用
test_questions = [
    "怎么在有序数组里做二分搜索？",
    "怎样用 transformers 加载模型？",
    "如何关闭 PyTorch 的梯度？"
]

for uq in test_questions:
    hits = retrieve_answer(uq, top_k=1)
    hit = hits[0]
    print(f"用户输入：{uq}")
    print(f"匹配问句：{hit['matched_question']}")
    print(f"答案      ：{hit['answer']}")
    print(f"相似度    ：{hit['score']:.4f}")
    print("-" * 50)

```

---

## 七、总结对比

- 计算效率 & 内存：TF–IDF 优势明显  
- 表示能力 & 下游性能：BERT 优势明显  
- 应用场景：  
  - 轻量检索、特征工程 → TF–IDF  
  - 深度理解、复杂任务 → BERT  

请选择最符合业务需求与资源约束的方法来构建您的文本处理管道。
