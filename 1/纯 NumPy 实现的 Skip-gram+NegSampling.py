import re
import numpy as np
import random
import pickle
from collections import Counter
from typing import List, Tuple

# ---------------------------------------------------
# 0. 分词：英文单词保留整词，中文按字符分词
# ---------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    用正则把英文单词、数字作为整体，中文汉字按单字符分词。
    其它字符（标点、空白）被忽略。
    """
    pattern = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
    return pattern.findall(text)

# ---------------------------------------------------
# 1. 构建词表 & 负采样分布
# ---------------------------------------------------
def build_vocab(corpus: List[str], min_count: int = 1):
    tokens = []
    for sent in corpus:
        tokens += tokenize(sent)
    vocab_counter = Counter(tokens)
    # 只保留出现 >= min_count 的词
    words = [w for w, c in vocab_counter.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}
    freq = np.array([vocab_counter[w] for w in words], dtype=np.float32)
    # unigram^0.75 负采样分布
    prob = freq ** 0.75
    prob /= prob.sum()
    return words, word2idx, idx2word, freq, prob

# ---------------------------------------------------
# 2. 产生 (target, context) 对
# ---------------------------------------------------
def generate_pairs(corpus: List[str],
                   word2idx: dict,
                   window_size: int) -> List[Tuple[int,int]]:
    pairs = []
    for sent in corpus:
        toks = tokenize(sent)
        ids = [word2idx[w] for w in toks if w in word2idx]
        for i, tid in enumerate(ids):
            start = max(0, i-window_size)
            end   = min(len(ids), i+window_size+1)
            for j in range(start, end):
                if i != j:
                    pairs.append((tid, ids[j]))
    return pairs

# ---------------------------------------------------
# 3. 负采样
# ---------------------------------------------------
def get_negative_samples(pos_idx: int,
                         vocab_size: int,
                         neg_size: int,
                         neg_prob: np.ndarray) -> List[int]:
    negs = []
    while len(negs) < neg_size:
        x = np.random.choice(vocab_size, p=neg_prob)
        if x != pos_idx:
            negs.append(x)
    return negs

# ---------------------------------------------------
# 4. Sigmoid
# ---------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------------------------------------------
# 5. 训练主函数
# ---------------------------------------------------
def train_sgns(corpus: List[str],
               embedding_dim: int = 100,
               window_size: int = 2,
               min_count: int = 1,
               neg_sample_size: int = 5,
               lr: float = 0.025,
               epochs: int = 5,
               verbose: bool = True):
    # 1) 构建词表
    words, word2idx, idx2word, freq, neg_prob = build_vocab(corpus, min_count)
    V = len(words)
    # 2) 初始化嵌入矩阵
    W_in  = (np.random.rand(V, embedding_dim) - 0.5) / embedding_dim
    W_out = (np.random.rand(V, embedding_dim) - 0.5) / embedding_dim
    # 3) 构建训练对
    pairs = generate_pairs(corpus, word2idx, window_size)

    # 4) 迭代训练
    for epoch in range(1, epochs+1):
        loss = 0.0
        random.shuffle(pairs)
        for target, context in pairs:
            v_t = W_in[target]      # (D,)
            u_c = W_out[context]    # (D,)

            # 正样本
            score_pos = sigmoid(np.dot(u_c, v_t))
            grad_pos = 1 - score_pos

            # 负样本
            negs = get_negative_samples(context, V, neg_sample_size, neg_prob)
            u_negs = W_out[negs]                   # (K,D)
            score_negs = sigmoid(-u_negs.dot(v_t)) # (K,)
            grad_negs = 1 - score_negs

            # 更新 W_out
            W_out[context] += lr * grad_pos * v_t
            for k, neg in enumerate(negs):
                W_out[neg] += lr * (-grad_negs[k]) * v_t

            # 更新 W_in
            grad_in = grad_pos * u_c + np.sum((-grad_negs[:,None]) * u_negs, axis=0)
            W_in[target] += lr * grad_in

            # 累计损失
            loss += -np.log(score_pos) - np.sum(np.log(score_negs))

        if verbose:
            print(f"Epoch {epoch}/{epochs}  Loss={loss:.4f}")

    # 5) 训练结束后 L2 规一化 W_in，方便余弦相似度直接 dot
    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    W_in = W_in / np.clip(norms, 1e-8, None)

    return {
        'words': words,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'W_in': W_in
    }

# ---------------------------------------------------
# 6. most_similar（余弦相似度检索）
# ---------------------------------------------------
def most_similar(model: dict, query: str, topn: int = 10):
    w2i = model['word2idx']
    i2w = model['idx2word']
    W = model['W_in']

    if query not in w2i:
        raise KeyError(f"'{query}' 不在词表中")
    idx = w2i[query]
    vec = W[idx]  # 单位向量

    sims = W.dot(vec)    # 直接 dot = cos sim
    sims[idx] = -1.0     # 排除自身
    best = np.argsort(-sims)[:topn]
    return [(i2w[i], float(sims[i])) for i in best]

# ---------------------------------------------------
# 7. 保存 / 加载
# ---------------------------------------------------
def save_model(model: dict, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

# ===================================================
# 使用示例
# ===================================================
if __name__ == '__main__':
    corpus = [
        "I love natural language processing",
        "自然 语言 处理 很 有趣",
        "Deep learning 推动 了 NLP 的 进展",
        "我 爱 Python 编程",
        "Language models are very powerful"
    ]

    model = train_sgns(
        corpus,
        embedding_dim=50,
        window_size=2,
        min_count=1,
        neg_sample_size=5,
        lr=0.05,
        epochs=200,
        verbose=True
    )

    print("\n与“自然”最相似：")
    for w,s in most_similar(model, "自然", topn=5):
        print(f"{w:>6}  {s:.4f}")

    print("\n与“language”最相似：")
    for w,s in most_similar(model, "language", topn=5):
        print(f"{w:>10}  {s:.4f}")

    # 保存与加载
    save_model(model, "w2v.pkl")
    m2 = load_model("w2v.pkl")
    print("\n[加载后] 与“学习”最相似：")
    for w,s in most_similar(m2, "学习", topn=5):
        print(f"{w:>6}  {s:.4f}")
