# -*- coding: utf-8 -*-
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 示例问答知识库
qa_pairs = [
    {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
    {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
    {"question": "如何使用Python进行数据分析？", "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。"},
    {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
    {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"}
]

# 从知识库提取所有问题列表
questions = [pair["question"] for pair in qa_pairs]

# 构建 TF-IDF 模型（用于 TF-IDF 相关算法）
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def compute_lcs_length(s1, s2):
    """
    计算两个字符串的最长公共子序列长度（LCS），利用动态规划算法
    """
    n = len(s1)
    m = len(s2)
    dp = [[0]*(m + 1) for _ in range(n + 1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def lcs_ratio(s1, s2):
    """
    计算 LCS 相似度百分比：
      ratio = (最长公共子序列长度 * 2) / (len(s1) + len(s2)) * 100
    """
    if not s1 and not s2:
        return 100
    lcs_len = compute_lcs_length(s1, s2)
    ratio = (lcs_len * 2) / (len(s1) + len(s2)) * 100
    return ratio

def search_by_edit_distance(user_question, threshold=50):
    """
    利用编辑距离（采用 fuzzywuzzy 的 fuzz.ratio）筛选候选问题，
    如果相似度低于 threshold，则不予考虑；否则返回最相似的问题索引
    """
    best_idx = -1
    best_score = 0
    for i, q in enumerate(questions):
        score = fuzz.ratio(user_question, q)
        # print(f"编辑距离比对: {q} 得分 {score}")
        if score >= threshold and score > best_score:
            best_score = score
            best_idx = i
    # 如果未找到达到阈值的候选，则取全局最高
    if best_idx == -1:
        for i, q in enumerate(questions):
            score = fuzz.ratio(user_question, q)
            if score > best_score:
                best_score = score
                best_idx = i
    return best_idx

def search_by_lcs(user_question, threshold=50):
    """
    利用最长公共子序列相似度计算匹配，同样要求相似度超过 threshold，
    返回最优匹配问题索引
    """
    best_idx = -1
    best_score = 0
    for i, q in enumerate(questions):
        score = lcs_ratio(user_question, q)
        # print(f"LCS 比对: {q} 得分 {score}")
        if score >= threshold and score > best_score:
            best_score = score
            best_idx = i
    # 如果没有候选项达到 threshold，则选择全局最高
    if best_idx == -1:
        for i, q in enumerate(questions):
            score = lcs_ratio(user_question, q)
            if score > best_score:
                best_score = score
                best_idx = i
    return best_idx

def search_by_tfidf(user_question):
    """
    利用 TF-IDF 模型和余弦相似度计算匹配，返回最优匹配问题索引
    """
    user_vec = vectorizer.transform([user_question])
    cosine_similarities = (tfidf_matrix * user_vec.T).toarray().ravel()
    best_idx = int(np.argmax(cosine_similarities))
    return best_idx

def get_best_answer(user_question, method="tfidf", threshold=50):
    """
    根据用户输入的问题以及指定的算法模式，返回最佳答案。
    
    参数：
      user_question - 用户输入的问题字符串
      method - 匹配算法，可以是 "edit_distance"、"lcs" 或 "tfidf"
      threshold - 针对 edit_distance 和 lcs 模式的初筛相似度阈值（0-100）
    """
    if method == "edit_distance":
        idx = search_by_edit_distance(user_question, threshold=threshold)
    elif method == "lcs":
        idx = search_by_lcs(user_question, threshold=threshold)
    elif method == "tfidf":
        idx = search_by_tfidf(user_question)
    else:
        raise ValueError("不支持的算法参数，请选择 edit_distance、lcs 或 tfidf")
    
    # 返回匹配到的回答
    return qa_pairs[idx]["answer"]

if __name__ == "__main__":
    print("欢迎使用问答系统！")
    print("请先选择匹配算法：")
    print("  1 - 编辑距离 (edit_distance)")
    print("  2 - 最长公共子序列 (lcs)")
    print("  3 - TF-IDF+余弦相似度 (tfidf)")
    
    mode_input = input("请输入匹配算法编号或名称：").strip()
    method = "tfidf"   # 默认使用 tfidf
    if mode_input in ["1", "edit_distance"]:
        method = "edit_distance"
    elif mode_input in ["2", "lcs"]:
        method = "lcs"
    elif mode_input in ["3", "tfidf"]:
        method = "tfidf"
    else:
        print("输入不合法，使用默认的 tfidf 模式。")
    
    print("匹配算法模式：", method)
    print("请输入问题，输入 'quit' 退出。")
    
    while True:
        user_input = input("用户：").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("感谢使用，再见！")
            break
        answer = get_best_answer(user_input, method=method, threshold=50)
        print("系统：", answer)
