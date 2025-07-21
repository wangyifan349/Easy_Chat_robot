# TF–IDF 与 BERT 精确对比

本文档分别从数学定义、模型结构、特征维度、计算效率、可解释性和典型应用等方面对 TF–IDF 与 BERT 进行详细对比，并提供基于 Hugging Face 社区微调模型的完整示例代码，帮助读者准确理解两者异同。

## TF–IDF

TF–IDF（Term Frequency–Inverse Document Frequency）是一种经典的文本向量化方法。给定语料库中文档总数 N 和词项 t 的文档频率 df(t)，逆文档频率定义为  
IDF(t)=log⁡(N/(1+df(t)))。文档 d 中词 t 的词频 TF(t,d) 通常取该词在 d 中的出现次数或其归一化形式。TF–IDF 权重计算为 TF(t,d)×IDF(t)。将每个文档表示为一个在词汇表维度上稀疏的实数向量，可直接用于余弦相似度计算、线性分类器或聚类算法。TF–IDF 不保留词序、上下文或深层语义，其优势在于计算开销小、需要的内存与语料规模线性相关、结果可解释性强。

下面的 Python 代码使用 scikit-learn 完整地加载文本、构建 TF–IDF 矩阵并示范查询文档与文档之间的相似度计算。请确保提前安装了 `scikit-learn` 与 `numpy`。

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "机器学习可以从数据中自动提取模型。",
    "深度学习是机器学习的一个分支，使用多层神经网络。",
    "TF IDF 是一种基于词频和逆文档频率的文本表示方法。"
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, norm='l2')
tfidf_matrix = vectorizer.fit_transform(documents)
query = "深度神经网络模型"
query_vec = vectorizer.transform([query])
similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
ranking = np.argsort(-similarities)

print("查询句子：", query)
for idx in ranking:
    print(f"文档 {idx}，相似度 {similarities[idx]:.4f}：{documents[idx]}")
```

## BERT

BERT（Bidirectional Encoder Representations from Transformers）基于 Transformer 编码器结构，在输入层将每个 token 的词嵌入与一个可学习的位置编码相加，借助多头自注意力机制同时关注序列中所有 token，从而捕捉词序和全局上下文依赖。BERT 在预训练阶段采用掩码语言模型（Masked Language Modeling）和下一句预测（Next Sentence Prediction）任务，对海量文本进行双向编码，得到上下文相关的稠密表示。下游微调时，在预训练模型之上插入任务特定的输出层，通过反向传播优化所有参数。BERT 的优势在于能够捕捉深层语义和长距离依赖，生成的向量维度通常为 768 或 1024，但它的预训练和推理计算成本较高，结果的可解释性低于 TF–IDF。

以下示例使用 Hugging Face 社区已经微调的多语言情感分析模型 `nlptown/bert-base-multilingual-uncased-sentiment`。示例代码完整展示了从文本编码、模型推理到提取句子向量的过程。请确保安装了 `transformers` 与 `torch`。

```python
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

texts = [
    "我非常喜欢这次模型的表现！",
    "这个示例的效果并不理想。"
]

encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**encoded)
last_hidden_state = outputs.last_hidden_state

cls_vectors = last_hidden_state[:, 0, :]
mean_vectors = last_hidden_state.mean(dim=1)

print("每条文本的 [CLS] 向量维度：", cls_vectors.shape)
print("每条文本的平均池化向量维度：", mean_vectors.shape)
```

## 精确对比

TF–IDF 直接计算词频与语料规模相关的权重，不保留词序或上下文，适合检索和基于线性模型的特征工程；
BERT 通过位置编码与自注意力机制生成上下文敏感的稠密表示，能够捕捉深层语义和长距离依赖，适合需要语言理解能力的分类、序列标注、问答和生成任务。
TF–IDF 在计算性能、内存占用和可解释性上更具优势，BERT 在表示能力和下游任务性能上更具优势。
根据具体业务需求与资源约束选择合适的方法。
