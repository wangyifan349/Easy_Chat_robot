import re
import numpy as np
import random
import pickle
from collections import Counter

# ----------------------------------------
# 1）中英文混合分词函数
# ----------------------------------------
def tokenize(text):
    """
    将一段文本分成“英文单词（或数字）”和“每个中文汉字”。
    其它字符（空格、标点）会被忽略。
    """
    pattern = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
    # re.findall 返回列表
    tokens = pattern.findall(text)
    return tokens

# ----------------------------------------
# 2）建立词表和负采样分布
# ----------------------------------------
def build_vocab(corpus, min_count):
    """
    输入：
      corpus: list of str，每条 str 可中英文混合
      min_count: 只保留出现次数 >= min_count 的 token
    输出：
      words:   词表列表，索引即 word_id
      word2id: 词到 id 的映射
      id2word: id 到词的映射
      neg_prob: 用于负采样的概率分布（numpy 数组）
    """
    # 2.1 收集所有 token
    all_tokens = []
    for sentence in corpus:
        tokens = tokenize(sentence)
        for w in tokens:
            all_tokens.append(w)

    # 2.2 统计词频
    counter = Counter()
    for w in all_tokens:
        counter[w] += 1

    # 2.3 建词表（过滤低频）
    words = []
    for w, freq in counter.items():
        if freq >= min_count:
            words.append(w)

    # 2.4 建映射
    word2id = {}
    id2word = {}
    for idx, w in enumerate(words):
        word2id[w] = idx
        id2word[idx] = w

    # 2.5 构造负采样分布 unigram^0.75
    freq_list = []
    for w in words:
        freq_list.append(counter[w])
    freq_array = np.array(freq_list, dtype=np.float32)

    # 先取 0.75 次方，再归一化
    prob = np.power(freq_array, 0.75)
    prob = prob / prob.sum()

    return words, word2id, id2word, prob

# ----------------------------------------
# 3）生成正样本对 (target, context)
# ----------------------------------------
def generate_pairs(corpus, word2id, window_size):
    """
    遍历每个句子，按滑动窗口生成 (target_id, context_id) 样本对
    """
    pairs = []
    for sentence in corpus:
        tokens = tokenize(sentence)
        # 先把句子转成 id 列表，忽略不在词表里的词
        id_list = []
        for w in tokens:
            if w in word2id:
                id_list.append(word2id[w])

        # i 位置为 target，j 位置为 context
        length = len(id_list)
        for i in range(length):
            target_id = id_list[i]
            # 窗口区间
            left = i - window_size
            right = i + window_size
            if left < 0:
                left = 0
            if right >= length:
                right = length - 1

            for j in range(left, right + 1):
                if j == i:
                    continue
                context_id = id_list[j]
                pairs.append((target_id, context_id))
    return pairs

# ----------------------------------------
# 4）负采样函数
# ----------------------------------------
def get_negative_samples(pos_id, vocab_size, neg_size, neg_prob):
    """
    为一个正样本 context id 采 neg_size 个负样本（不能等于 pos_id）
    """
    negs = []
    while len(negs) < neg_size:
        sampled = np.random.choice(vocab_size, p=neg_prob)
        if sampled != pos_id:
            negs.append(sampled)
    return negs

# ----------------------------------------
# 5）Sigmoid
# ----------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ----------------------------------------
# 6）训练 Skip-gram + Negative Sampling
# ----------------------------------------
def train_sgns(
    corpus,
    embedding_dim=100,
    window_size=2,
    min_count=1,
    neg_sample_size=5,
    lr=0.025,
    epochs=5,
    verbose=True
):
    # ---- 6.1 建词表和负采样分布
    words, word2id, id2word, neg_prob = build_vocab(corpus, min_count)
    vocab_size = len(words)

    # ---- 6.2 初始化输入/输出嵌入矩阵
    W_in  = np.random.rand(vocab_size, embedding_dim).astype(np.float32)
    W_in  = (W_in - 0.5) / embedding_dim
    W_out = np.random.rand(vocab_size, embedding_dim).astype(np.float32)
    W_out = (W_out - 0.5) / embedding_dim

    # ---- 6.3 生成所有正样本对
    pairs = generate_pairs(corpus, word2id, window_size)

    # ---- 6.4 迭代训练
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        # 打乱样本顺序
        random.shuffle(pairs)

        # 遍历每个 (t, c)
        for (t_id, c_id) in pairs:
            v_t = W_in[t_id]      # 目标词向量 (D,)
            u_c = W_out[c_id]     # 正样本上下文向量 (D,)

            # 正样本得分与梯度
            score_pos = sigmoid(np.dot(u_c, v_t))
            grad_pos  = 1.0 - score_pos

            # 负样本
            neg_ids = get_negative_samples(c_id, vocab_size, neg_sample_size, neg_prob)
            # 收集负样本向量
            neg_vecs = []
            for nid in neg_ids:
                neg_vecs.append(W_out[nid])
            neg_vecs = np.stack(neg_vecs, axis=0)  # (K, D)

            # 负样本得分与梯度
            scores_neg = sigmoid(- np.dot(neg_vecs, v_t))  # (K,)
            grad_negs  = 1.0 - scores_neg                  # (K,)

            # ---- 更新 W_out
            # 正样本方向
            W_out[c_id] += lr * grad_pos * v_t
            # 每个负样本
            for k in range(len(neg_ids)):
                nid = neg_ids[k]
                W_out[nid] += lr * (- grad_negs[k]) * v_t

            # ---- 更新 W_in
            # grad_in = grad_pos * u_c + sum( -grad_negs[k] * neg_vecs[k] )
            grad_in = grad_pos * u_c
            for k in range(len(neg_ids)):
                grad_in += (- grad_negs[k]) * neg_vecs[k]
            W_in[t_id] += lr * grad_in

            # 累计损失（可选监控）
            total_loss += - np.log(score_pos)
            for s in scores_neg:
                total_loss += - np.log(s)

        # 每轮打印损失
        if verbose:
            print(f"Epoch {epoch}/{epochs}  Loss={total_loss:.4f}")

    # ---- 6.5 训练完毕后做 L2 规范化，方便后续用内积算余弦
    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    W_in = W_in / np.clip(norms, 1e-8, None)

    # 返回模型字典
    model = {
        'words': words,
        'word2id': word2id,
        'id2word': id2word,
        'W_in': W_in.astype(np.float32)
    }
    return model

# ----------------------------------------
# 7）Faiss 索引与检索
# ----------------------------------------
def build_faiss_index(W, use_gpu=False):
    """
    W: (V, D) 单位向量矩阵，dtype=float32
    use_gpu: 是否用 GPU（需安装 faiss-gpu）
    返回：faiss.Index 对象
    """
    import faiss
    V, D = W.shape
    # 用内积搜索，向量已规范化即等同于余弦
    index = faiss.IndexFlatIP(D)
    if use_gpu:
        # 将 index 放到 GPU 上
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(W)   # 把所有向量添加进去
    return index

def faiss_most_similar(model, index, query, topn=10):
    """
    用 Faiss 查 query 的近邻 topn
    """
    word2id = model['word2id']
    id2word = model['id2word']
    W = model['W_in']

    # query 必须在词表中
    if query not in word2id:
        raise KeyError(f"'{query}' 不在词表里")

    q_id = word2id[query]
    # 取出 query 向量并 reshape 成 (1, D)
    q_vec = W[q_id].reshape(1, -1).astype(np.float32)

    # search 会返回 topn+1（包含自己），因此请求 topn+1 个
    sims, ids = index.search(q_vec, topn + 1)
    sims = sims[0]
    ids  = ids[0]

    results = []
    for i, wid in enumerate(ids):
        if wid == q_id:
            continue
        results.append((id2word[wid], float(sims[i])))
        if len(results) >= topn:
            break

    return results
# ----------------------------------------
# 8）模型保存 / 加载
# ----------------------------------------
def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
# ========================================
# 9）使用示例
# ========================================
if __name__ == '__main__':
    # 准备一个混合中英文的小语料
    corpus = [
        "I love natural language processing",
        "自然 语言 处理 很 有趣",
        "Deep learning 推动 了 NLP 的 进展",
        "我 爱 Python 编程",
        "Language models are very powerful"
    ]

    # 训练词向量
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

    # 用 Faiss 建立索引（CPU）
    faiss_index = build_faiss_index(model['W_in'], use_gpu=False)

    # 检索示例
    print("\nFaiss 检索：与“自然”最相近词")
    results = faiss_most_similar(model, faiss_index, "自然", topn=5)
    for w, sim in results:
        print(f"{w:>6}  sim={sim:.4f}")

    print("\nFaiss 检索：与“language”最相近词")
    results = faiss_most_similar(model, faiss_index, "language", topn=5)
    for w, sim in results:
        print(f"{w:>10}  sim={sim:.4f}")

    # 保存 & 加载
    save_model(model, "w2v_faiss.pkl")
    loaded = load_model("w2v_faiss.pkl")

    # 重新建索引并检索
    new_index = build_faiss_index(loaded['W_in'], use_gpu=False)
    print("\n加载后检索：与“学习”最相近词")
    results = faiss_most_similar(loaded, new_index, "学习", topn=5)
    for w, sim in results:
        print(f"{w:>6}  sim={sim:.4f}")
