import numpy as np
from collections import Counter
import re

# 定义最长公共子序列函数
def lcs(X, Y):
    # 获取两个序列的长度
    m = len(X)
    n = len(Y)
    # 创建一个二维数组来存储最长公共子序列的长度
    L = [[0] * (n + 1) for _ in range(m + 1)]
    # 遍历两个序列
    for i in range(m + 1):
        for j in range(n + 1):
            # 如果任一序列为空，则公共子序列的长度为0
            if i == 0 or j == 0:
                L[i][j] = 0
            # 如果当前字符相同，则在之前的基础上+1
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            # 否则选择前面两者中的较大值
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    # 利用动态规划矩阵回溯获得 LCS
    index = L[m][n]
    lcs_list = [""] * (index + 1)
    lcs_list[index] = ""
    
    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs_list[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return "".join(lcs_list)

# 定义编辑距离函数
def edit_distance(X, Y):
    m = len(X)
    n = len(Y)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # 填充DP数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if X[i - 1] == Y[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # 删除
                dp[i][j - 1] + 1,       # 插入
                dp[i - 1][j - 1] + cost # 替换
            )
    
    return dp[m][n]

# 定义TF-IDF函数
def tf_idf(documents):
    # 构建所有文档的词汇表
    vocabulary = set()
    for document in documents:
        vocabulary.update(document.split())
    
    # 对词汇进行排序，保证向量转换时顺序一致
    vocabulary = sorted(vocabulary)
    
    # 预统计各单词在多少文档中出现（文档频率）
    N = len(documents)
    doc_freq = {}
    for word in vocabulary:
        doc_freq[word] = sum(1 for doc in documents if word in doc.split())
    
    # 计算TF-IDF值
    tf_idf_values = []
    for document in documents:
        tf_idf_value = {}
        words = document.split()
        word_count = len(words)
        counter = Counter(words)
        for word in vocabulary:
            tf = counter[word] / word_count if word_count > 0 else 0
            # 防止除零错误或idf为0，可在分母加1平滑，但这里假定每个词至少出现一次
            idf = np.log(N / (doc_freq[word] if doc_freq[word] > 0 else 1))
            tf_idf_value[word] = tf * idf
        tf_idf_values.append(tf_idf_value)
    
    return tf_idf_values, vocabulary

# 定义余弦相似度函数
def cosine_similarity(X, Y):
    dot_product = np.dot(X, Y)
    magnitude_X = np.linalg.norm(X)
    magnitude_Y = np.linalg.norm(Y)
    if magnitude_X == 0 or magnitude_Y == 0:
        return 0.0
    return dot_product / (magnitude_X * magnitude_Y)

# 定义利用TF-IDF计算文档向量并计算余弦相似度的函数
def tf_idf_cosine_similarity(documents):
    tf_idf_values, vocabulary = tf_idf(documents)
    
    # 将每个文档的TF-IDF字典转换为向量（按照 vocabulary 顺序）
    vectors = []
    for tf_idf_value in tf_idf_values:
        vector = [tf_idf_value[word] for word in vocabulary]
        vectors.append(np.array(vector))
    
    n = len(vectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])
    
    return similarity_matrix

# 新增函数：比较两个句子的相似性
def compare_sentences_similarity(sentence1, sentence2):
    """
    利用TF-IDF和余弦相似度比较两个句子的相似性
    返回一个0~1之间的相似度值，值越大表示越相似。
    """
    # 将两个句子作为两个文档处理
    documents = [sentence1, sentence2]
    similarity_matrix = tf_idf_cosine_similarity(documents)
    # 返回相似度矩阵中两个句子之间的相似度值
    return similarity_matrix[0, 1]

# 示例测试
if __name__ == "__main__":
    # 测试最长公共子序列
    seq1 = "ABCBDAB"
    seq2 = "BDCABC"
    print("Longest Common Subsequence:", lcs(seq1, seq2))
    
    # 测试编辑距离
    s1 = "kitten"
    s2 = "sitting"
    print("Edit Distance:", edit_distance(s1, s2))
    
    # 测试TF-IDF和余弦相似度（多文档情况）
    docs = [
        "this is a sample document",
        "this document is a sample",
        "sample document is sample"
    ]
    
    sim_matrix = tf_idf_cosine_similarity(docs)
    print("TF-IDF Cosine Similarity Matrix:")
    print(sim_matrix)
    
    # 测试对比两个句子的相似性
    sentence1 = "今天的天气很好，适合出去散步。"
    sentence2 = "今天天气晴朗，适合出门走走。"
    similarity = compare_sentences_similarity(sentence1, sentence2)
    print("Sentence Similarity:", similarity)
