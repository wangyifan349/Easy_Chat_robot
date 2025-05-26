# -*- coding: utf-8 -*-
"""
综合问答系统

预设了多个问答对，并支持三种匹配算法：
  - 编辑距离 (利用 fuzzywuzzy 库中的 fuzz.ratio)
  - 最长公共子序列 (LCS)
  - TF-IDF 向量化后计算余弦相似度
为了获得更稳定的匹配结果，提供了融合多种算法的综合匹配方案，
融合逻辑主要为：
  1. 将所有匹配算法返回的相似度分数统一归一化到 0～100 的分数。
  2. 根据设定权重对各个算法的分数进行加权求和。
  3. 返回综合得分最高的候选问题对应的回答。
根据实际情况调整各个算法的权重和匹配阈值。
"""

from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 示例问答数据（知识库）
qa_pairs = [
    {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
    {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
    {"question": "如何使用Python进行数据分析？", "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。"},
    {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
    {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"}
]

# 提取所有问题，方便后续遍历处理
questions = [pair["question"] for pair in qa_pairs]

# 构建 TF-IDF 模型，用于计算 TF-IDF 向量及余弦相似度
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def compute_lcs_length(s1, s2):
    """
    计算两个字符串的最长公共子序列长度（LCS），使用动态规划算法。
    参数：
      s1, s2: 算法输入的两个字符串
    返回：
      dp[n][m]：两个字符串的最长公共子序列长度
    """
    n = len(s1)
    m = len(s2)
    dp = [[0]*(m + 1) for _ in range(n + 1)]
    # 遍历两个字符串的每个字符，填充DP数组
    for i in range(1, n+1):
        for j in range(1, m+1):
            # 当当前字符相等时，增加公共子序列长度
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            # 不相等时，取先前计算结果的最大值
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def lcs_ratio(s1, s2):
    """
    根据最长公共子序列长度计算相似度百分比：
      ratio = (2 * LCS长度) / (len(s1) + len(s2)) * 100
    参数：
      s1, s2: 输入的两个字符串
    返回：
      相似度百分比（0～100）
    """
    # 当两个字符串都为空时，认为相似度为100%
    if not s1 and not s2:
        return 100
    lcs_len = compute_lcs_length(s1, s2)
    ratio = (lcs_len * 2) / (len(s1) + len(s2)) * 100
    return ratio

def tfidf_score(user_question, q):
    """
    计算 TF-IDF 模型下的问题相似度（余弦相似度），将结果归一化到 0～100 的分数范围。
    参数：
      user_question: 用户输入的问题
      q: 知识库中的候选问题
    返回：
      余弦相似度*100后的分数（0～100）
    """
    # 将用户输入和候选问题转换为TF-IDF向量
    user_vec = vectorizer.transform([user_question])
    q_vec = vectorizer.transform([q])
    # 计算余弦相似度
    cosine_score = (user_vec * q_vec.T).toarray()[0][0]
    # 乘100归一化到0~100区间
    return cosine_score * 100

def search_by_all_methods(user_question, threshold=50):
    """
    综合采用编辑距离、LCS和TF-IDF三种方法进行匹配，计算加权总分，并返回最高分候选的下标。
    参数：
      user_question: 用户输入的问题
      threshold: 融合匹配的最低阈值，只有超过此阈值的候选才有机会被选中（0~100）
    返回：
      得分最高的候选问题的索引
    """
    # 各算法权重设置（可根据实际效果调整）
    weight_edit = 0.4    # 编辑距离的权重
    weight_lcs = 0.3     # LCS的权重
    weight_tfidf = 0.3   # TF-IDF的权重

    best_idx = -1
    best_total_score = -1

    # 遍历知识库中的每个问题
    for i, q in enumerate(questions):
        # 调用 fuzzywuzzy 库计算编辑距离相似度，返回百分比（0~100）
        score_edit = fuzz.ratio(user_question, q)
        # 计算 LCS 相似度，返回百分比
        score_lcs = lcs_ratio(user_question, q)
        # 计算 TF-IDF 余弦相似度，归一化到0~100
        score_tfidf = tfidf_score(user_question, q)

        # 计算综合得分：各算法得分按照预设权重加权平均
        total_score = weight_edit * score_edit + weight_lcs * score_lcs + weight_tfidf * score_tfidf

        # 如果当前问题的总得分达到阈值且高于当前最佳，则更新最佳候选
        if total_score >= threshold and total_score > best_total_score:
            best_total_score = total_score
            best_idx = i

    # 若所有问题都未达到阈值，则依旧返回全局最高得分的候选答案
    if best_idx == -1:
        for i, q in enumerate(questions):
            score_edit = fuzz.ratio(user_question, q)
            score_lcs = lcs_ratio(user_question, q)
            score_tfidf = tfidf_score(user_question, q)
            total_score = weight_edit * score_edit + weight_lcs * score_lcs + weight_tfidf * score_tfidf
            if total_score > best_total_score:
                best_total_score = total_score
                best_idx = i

    return best_idx

def get_best_answer(user_question, method="tfidf", threshold=50):
    """
    根据用户输入问题及指定匹配算法，选择最佳答案。
    支持以下匹配模式：
      - "edit_distance": 仅采用编辑距离计算相似度
      - "lcs": 仅采用最长公共子序列计算相似度
      - "tfidf": 仅采用TF-IDF余弦相似度计算匹配
      - "all": 使用融合策略（综合三种方法）
    参数：
      user_question: 用户输入问题
      method: 匹配算法模式（默认 tfidf）
      threshold: 针对单一算法或者融合模式设定的相似度阈值（0~100）
    返回：
      匹配到的最佳答案字符串
    """
    if method == "edit_distance":
        # 采用编辑距离进行匹配
        best_idx = -1
        best_score = 0
        for i, q in enumerate(questions):
            score = fuzz.ratio(user_question, q)
            if score >= threshold and score > best_score:
                best_score = score
                best_idx = i
        # 若无达到阈值，则返回全局最高得分的候选
        if best_idx == -1:
            for i, q in enumerate(questions):
                score = fuzz.ratio(user_question, q)
                if score > best_score:
                    best_score = score
                    best_idx = i

    elif method == "lcs":
        # 采用最长公共子序列进行匹配
        best_idx = -1
        best_score = 0
        for i, q in enumerate(questions):
            score = lcs_ratio(user_question, q)
            if score >= threshold and score > best_score:
                best_score = score
                best_idx = i
        # 若没有候选问题达到阈值，则返回最高得分问题
        if best_idx == -1:
            for i, q in enumerate(questions):
                score = lcs_ratio(user_question, q)
                if score > best_score:
                    best_score = score
                    best_idx = i

    elif method == "tfidf":
        # 采用 TF-IDF + 余弦相似度进行匹配
        best_idx = -1
        best_score = -1
        for i, q in enumerate(questions):
            score = tfidf_score(user_question, q)
            if score > best_score:
                best_score = score
                best_idx = i

    elif method == "all":
        # 使用融合策略，综合三种算法的得分进行匹配
        best_idx = search_by_all_methods(user_question, threshold=threshold)

    else:
        # 如果指定了不支持的匹配算法，抛出异常
        raise ValueError("不支持的算法参数，请选择 edit_distance、lcs、tfidf 或 all")
    
    # 返回最佳匹配问题对应的答案
    return qa_pairs[best_idx]["answer"]

if __name__ == "__main__":
    # 欢迎提示以及算法选择菜单
    print("欢迎使用问答系统！")
    print("请先选择匹配算法：")
    print("  1 - 编辑距离 (edit_distance)")
    print("  2 - 最长公共子序列 (lcs)")
    print("  3 - TF-IDF+余弦相似度 (tfidf)")
    print("  4 - 综合匹配 (all) —— 融合多种算法")
    
    mode_input = input("请输入匹配算法编号或名称：").strip()
    method = "tfidf"   # 默认选项为 tfidf
    if mode_input in ["1", "edit_distance"]:
        method = "edit_distance"
    elif mode_input in ["2", "lcs"]:
        method = "lcs"
    elif mode_input in ["3", "tfidf"]:
        method = "tfidf"
    elif mode_input in ["4", "all"]:
        method = "all"
    else:
        print("输入不合法，使用默认的 tfidf 模式。")
    
    print("匹配算法模式：", method)
    print("请输入问题，输入 'quit' 退出。")
    
    # 主循环：不断收到用户输入的问题，匹配并返回答案
    while True:
        user_input = input("用户：").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("感谢使用，再见！")
            break
        # 获取最佳匹配答案
        answer = get_best_answer(user_input, method=method, threshold=50)
        print("系统：", answer)
