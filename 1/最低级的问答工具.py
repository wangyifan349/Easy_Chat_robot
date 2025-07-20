#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys                               # 系统模块，用于退出
import random                            # 随机选择回答
import jieba                             # 中文分词
import numpy as np                      # 数值计算
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 向量化
from sklearn.metrics.pairwise import cosine_similarity        # 余弦相似度


# -------------------------------------------------------------------
# 工具函数：LCS、编辑距离、中文分词
# -------------------------------------------------------------------

def longest_common_subsequence(a, b):
    """
    计算两个字符串 a, b 的最长公共子序列长度（LCS）。
    """
    n = len(a); m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def edit_distance(a, b):
    """
    计算字符串 a -> b 的最小编辑距离（Levenshtein 距离）。
    """
    n = len(a); m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i        # 删除成本
    for j in range(m + 1):
        dp[0][j] = j        # 插入成本

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v1 = dp[i - 1][j] + 1       # 删除
            v2 = dp[i][j - 1] + 1       # 插入
            v3 = dp[i - 1][j - 1] + cost  # 替换或匹配
            dp[i][j] = min(v1, v2, v3)
    return dp[n][m]


def jieba_tokenizer(text):
    """
    使用 jieba 对中文进行分词，返回词列表。
    """
    return jieba.lcut(text)


# -------------------------------------------------------------------
# 问答知识库 & TF-IDF 预处理
# -------------------------------------------------------------------

qa_dict = {
    "你叫什么名字？": ["我叫小问问。", "我是智能小问，随时为你服务。"],
    "打印 Hello World": [
        "def main():\n"
        "    print(\"Hello World\")\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()"
    ],
    "示例循环": ["for i in range(5):\n    print(i)"],
    "最大公约数": [
        "def gcd(a, b):\n"
        "    while b != 0:\n"
        "        a, b = b, a % b\n"
        "    return a"
    ],
    "今天天气怎么样？": [
        "抱歉，我无法获取实时天气，请去天气网站查询。",
        "我不知道外面的天气，但希望你今天心情晴朗！"
    ],
    "最长公共子序列是什么？": [
        "最长公共子序列是两个序列中同时为子序列的最长序列。",
        "LCS 是一类经典的字符串动态规划问题。"
    ],
    "什么是编辑距离？": [
        "编辑距离是一个字符串变成另一个字符串所需的最少编辑操作次数。",
        "Levenshtein 距离度量字符串相似度。"
    ],
}

questions = list(qa_dict.keys())         # 所有问题列表
vectorizer = TfidfVectorizer(
    tokenizer=jieba_tokenizer,           # 使用 jieba 分词
    lowercase=False                      # 保持原有大小写
)
tfidf_matrix = vectorizer.fit_transform(questions)  # 问题集 TF-IDF 向量化


# -------------------------------------------------------------------
# 四种匹配方法：LCS、编辑距离、Cosine、混合
# -------------------------------------------------------------------

def best_match_by_lcs(user_q):
    best_q, best_len = None, -1
    for q in questions:
        l = longest_common_subsequence(user_q, q)
        if l > best_len:
            best_len, best_q = l, q
    return best_q


def best_match_by_edit(user_q):
    best_q, best_dist = None, None
    for q in questions:
        d = edit_distance(user_q, q)
        if best_dist is None or d < best_dist:
            best_dist, best_q = d, q
    return best_q


def best_match_by_cosine(user_q):
    query_vec = vectorizer.transform([user_q])          # 用户问题 TF-IDF
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]  # 与所有问题余弦相似度
    best_index, best_score = 0, sims[0]
    for i in range(1, len(sims)):
        if sims[i] > best_score:
            best_score, best_index = sims[i], i
    return questions[best_index]


def best_match_by_hybrid(user_q):
    # LCS 分数归一化
    lcs_scores = []
    for q in questions:
        lcs_scores.append(longest_common_subsequence(user_q, q))
    maxlen = max(len(user_q), max(len(q) for q in questions), 1)
    for i in range(len(lcs_scores)):
        lcs_scores[i] /= maxlen

    # Cosine 分数
    query_vec = vectorizer.transform([user_q])
    cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # 50% LCS + 50% Cosine
    hybrid_scores = []
    for i in range(len(questions)):
        hybrid_scores.append(0.5 * lcs_scores[i] + 0.5 * cos_scores[i])

    best_index, best_score = 0, hybrid_scores[0]
    for i in range(1, len(hybrid_scores)):
        if hybrid_scores[i] > best_score:
            best_score, best_index = hybrid_scores[i], i
    return questions[best_index]


# -------------------------------------------------------------------
# 问答主流程
# -------------------------------------------------------------------

def answer_question(user_q, method):
    if method == 1:
        key = best_match_by_lcs(user_q)
    elif method == 2:
        key = best_match_by_edit(user_q)
    elif method == 3:
        key = best_match_by_cosine(user_q)
    elif method == 4:
        key = best_match_by_hybrid(user_q)
    else:
        return "未识别的匹配方法。"

    answers = qa_dict.get(key, [])
    if not answers:
        return "抱歉，我不清楚如何回答。"
    return random.choice(answers)  # 随机选一个答案


def main():
    print("欢迎使用智能问答系统！")
    print("请选择匹配方法：")
    print("  1. LCS（最长公共子序列）")
    print("  2. 编辑距离（Edit Distance）")
    print("  3. TF-IDF + 余弦相似度（Cosine）")
    print("  4. LCS + Cosine 混合（50%:50%）")

    choice = input("请输入方法编号 [1-4]：").strip()
    if choice not in ("1", "2", "3", "4"):
        print("输入无效，程序退出。")
        sys.exit(1)
    method = int(choice)

    print("输入“退出”或“exit”结束对话。")
    while True:
        user_q = input("\n你的问题：").rstrip()
        if not user_q:
            continue
        if user_q.lower() in ("退出", "exit", "quit"):
            print("再见！")
            break
        ans = answer_question(user_q, method)
        print("回答：")
        print(ans)  # 支持多行代码和文本输出

if __name__ == "__main__":
    main()
