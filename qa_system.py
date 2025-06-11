# -*- coding: utf-8 -*-
"""
基于多种匹配算法的中文问答系统
此模块实现了一个支持多种匹配算法的问答系统，包括：
  - 编辑距离（使用 fuzzywuzzy 的 fuzz.ratio）
  - 最长公共子序列（LCS）
  - 基于 TF-IDF 向量化的余弦相似度
并提供了一个加权融合的策略来合并所有方法的得分：
  1. 将所有相似度得分归一化到 [0, 100] 的范围。
  2. 计算加权总和。
  3. 返回得分最高的前 k 个候选问题的答案。
使用 jieba 进行中文分词。
系统支持编程问答，能正确格式化代码块输出
"""
# ----------------------
# 导入必要的库
# ----------------------
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# ----------------------
# 定义使用 jieba 的中文分词函数
# ----------------------
def chinese_tokenizer(text):
    """使用 jieba 对中文文本进行分词。"""
    return jieba.lcut(text)

# ----------------------
# 定义知识库：问答对
# ----------------------
qa_pairs = [
    {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
    {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
    {"question": "如何使用Python进行数据分析？", "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。"},
    {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
    {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"},
    # 编程相关的问答，使用三引号的多行字符串保存代码块
    {"question": "如何编写Python函数？", "answer": """你可以使用'def'关键词来定义一个Python函数。例如：

```python
def my_function(param1, param2):
    '''这个函数执行某些操作。'''
    result = param1 + param2
    return result
```
"""}
]

# ----------------------
# 从知识库中提取所有问题
# ----------------------
questions = []
for pair in qa_pairs:
    questions.append(pair["question"])

# ----------------------
# 使用自定义的中文分词器构建 TF-IDF 模型
# ----------------------
vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
vectorizer.fit(questions)

# ----------------------
# 最长公共子序列（LCS）函数
# ----------------------
def compute_lcs_length(s1, s2):
    """
    计算两个字符串之间的最长公共子序列的长度。

    参数：
        s1: 第一个字符串。
        s2: 第二个字符串。

    返回：
        LCS 的长度。
    """
    n = len(s1)
    m = len(s2)
    # 初始化 DP 表格，全为 0
    dp = []
    for _ in range(n + 1):
        dp.append([0] * (m + 1))
    # 填充 DP 表格
    for i in range(n):
        for j in range(m):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] +1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def lcs_ratio(s1, s2):
    """
    基于 LCS 计算相似度百分比：
      ratio = (2 * LCS_length) / (len(s1) + len(s2)) * 100
    参数：
        s1: 第一个字符串。
        s2: 第二个字符串。
    返回：
        [0, 100] 范围内的相似度百分比。
    """
    if not s1 and not s2:
        return 100
    lcs_len = compute_lcs_length(s1, s2)
    ratio = (lcs_len * 2) / (len(s1) + len(s2)) * 100
    return ratio
# ----------------------
# TF-IDF 相似度得分函数
# ----------------------
def tfidf_score(user_question, candidate_question):
    """
    计算用户问题和候选问题的 TF-IDF 向量之间的余弦相似度。
    参数：
        user_question: 用户输入的问题。
        candidate_question: 知识库中的候选问题。

    返回：
        余弦相似度乘以 100（范围 [0, 100]）。
    """
    user_vec = vectorizer.transform([user_question])
    candidate_vec = vectorizer.transform([candidate_question])
    cosine_score = (user_vec * candidate_vec.T).toarray()[0][0]
    return cosine_score * 100
# ----------------------
# 使用所有方法的加权融合匹配策略
# ----------------------
def search_by_all_methods(user_question, threshold=50, top_k=1):
    """
    结合编辑距离、LCS 和 TF-IDF 来计算加权总相似度得分，返回知识库中最匹配的前 k 个候选问题的索引。
    参数：
        user_question: 用户输入的问题。
        threshold: 考虑候选项的最低阈值（范围：0-100）。
        top_k: 返回最匹配的前 k 个结果。
    返回：
        最佳匹配问题的索引列表。
    """
    # 每种方法的权重设置
    weight_edit = 0.4    # 编辑距离的权重
    weight_lcs = 0.3     # LCS 的权重
    weight_tfidf = 0.3   # TF-IDF 的权重
    scores = []
    for i in range(len(questions)):
        candidate = questions[i]
        score_edit = fuzz.ratio(user_question, candidate)      # 编辑距离得分
        score_lcs = lcs_ratio(user_question, candidate)        # LCS 相似度得分
        score_tfidf = tfidf_score(user_question, candidate)    # TF-IDF 余弦得分
        total_score = weight_edit * score_edit + weight_lcs * score_lcs + weight_tfidf * score_tfidf
        if total_score >= threshold:
            scores.append((i, total_score))
    if not scores:
        # 如果没有候选项满足阈值，则使用所有得分
        for i in range(len(questions)):
            candidate = questions[i]
            score_edit = fuzz.ratio(user_question, candidate)
            score_lcs = lcs_ratio(user_question, candidate)
            score_tfidf = tfidf_score(user_question, candidate)
            total_score = weight_edit * score_edit + weight_lcs * score_lcs + weight_tfidf * score_tfidf
            scores.append((i, total_score))
    # 按得分从高到低排序
    scores.sort(key=lambda x: x[1], reverse=True)
    # 获取前 k 个候选项的索引
    top_indices = []
    for idx, score in scores[:top_k]:
        top_indices.append(idx)
    return top_indices
# ----------------------
# 获取最佳答案的函数
# ----------------------
def get_best_answers(user_question, method="tfidf", threshold=50, top_k=1):
    """
    使用指定的算法匹配用户问题，从知识库中获取最佳答案。
    支持的算法：
      - "edit_distance": 仅使用编辑距离进行匹配。
      - "lcs": 仅使用最长公共子序列进行匹配。
      - "tfidf": 仅使用 TF-IDF 余弦相似度进行匹配。
      - "all": 使用融合策略，合并所有三种方法。
    参数：
        user_question: 用户输入的问题。
        method: 使用的匹配算法（默认是 "tfidf"）。
        threshold: 单一或融合策略的相似度阈值（范围：0-100）。
        top_k: 返回最匹配的前 k 个答案。
    返回：
        最佳匹配答案的列表。
    """
    if method == "edit_distance":
        scores = []
        for i in range(len(questions)):
            candidate = questions[i]
            score = fuzz.ratio(user_question, candidate)
            if score >= threshold:
                scores.append((i, score))
        if not scores:
            for i in range(len(questions)):
                candidate = questions[i]
                score = fuzz.ratio(user_question, candidate)
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = []
        for idx, score in scores[:top_k]:
            top_indices.append(idx)
    elif method == "lcs":
        scores = []
        for i in range(len(questions)):
            candidate = questions[i]
            score = lcs_ratio(user_question, candidate)
            if score >= threshold:
                scores.append((i, score))
        if not scores:
            for i in range(len(questions)):
                candidate = questions[i]
                score = lcs_ratio(user_question, candidate)
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = []
        for idx, score in scores[:top_k]:
            top_indices.append(idx)
    elif method == "tfidf":
        scores = []
        for i in range(len(questions)):
            candidate = questions[i]
            score = tfidf_score(user_question, candidate)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = []
        for idx, score in scores[:top_k]:
            top_indices.append(idx)
    elif method == "all":
        top_indices = search_by_all_methods(user_question, threshold=threshold, top_k=top_k)
    else:
        raise ValueError("不支持的算法。请选择：edit_distance、lcs、tfidf 或 all")
    # 获取对应的答案
    best_answers = []
    for idx in top_indices:
        best_answers.append(qa_pairs[idx]["answer"])
    return best_answers
# ----------------------
# 带有编程支持的交互式对话循环
# ----------------------
def main():
    print("欢迎使用问答系统！")
    print("请选择匹配算法：")
    print("  1 - 编辑距离 (edit_distance)")
    print("  2 - 最长公共子序列 (lcs)")
    print("  3 - TF-IDF + 余弦相似度 (tfidf)")
    print("  4 - 加权融合匹配 (all)")
    mode_input = input("请输入匹配算法的编号或名称：").strip()
    if mode_input in ["1", "edit_distance"]:
        method = "edit_distance"
    elif mode_input in ["2", "lcs"]:
        method = "lcs"
    elif mode_input in ["3", "tfidf"]:
        method = "tfidf"
    elif mode_input in ["4", "all"]:
        method = "all"
    else:
        print("输入无效，默认使用 tfidf 模式。")
        method = "tfidf"
    top_k_input = input("请输入返回答案的数量（top k）：").strip()
    if top_k_input.isdigit() and int(top_k_input) > 0:
        top_k = int(top_k_input)
    else:
        print("输入无效，默认返回 1 个答案。")
        top_k = 1
    print("已选择匹配模式：", method)
    print("请输入您的问题（输入 'quit'、'exit' 或 '退出' 结束程序）。")
    while True:
        user_input = input("用户：").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("感谢您的使用，再见！")
            break

        # 获取最佳答案
        answers = get_best_answers(user_input, method=method, threshold=50, top_k=top_k)
        print("系统：")
        for answer in answers:
            print(answer)
            print("-" * 50)

if __name__ == "__main__":
    main()
