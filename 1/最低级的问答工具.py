#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Dict, List, Union

AnswerType = Union[str, str]  # 普通回答和代码都用 str，但代码用三引号多行

def longest_common_subsequence(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[n][m]

# 问答知识库：直接存储多行代码（真实换行）或单行文本
qa_dict: Dict[str, List[AnswerType]] = {
    "你叫什么名字？": [
        "我叫小问问。",
        "我是智能小问，随时为你服务。"
    ],
    "打印 Hello World": [
        """def main():
    print("Hello World")

if __name__ == "__main__":
    main()"""
    ],
    "示例循环": [
        """for i in range(5):
    print(i)"""
    ],
    "最大公约数": [
        """def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a"""
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

def best_match_by_edit(question: str) -> str:
    best_q, best_d = None, float('inf')
    for q in qa_dict:
        d = edit_distance(question, q)
        if d < best_d:
            best_d, best_q = d, q
    return best_q

def best_match_by_lcs(question: str) -> str:
    best_q, best_l = None, -1
    for q in qa_dict:
        l = longest_common_subsequence(question, q)
        if l > best_l:
            best_l, best_q = l, q
    return best_q

def answer_question(question: str, method: str = 'edit') -> str:
    """
    method = 'edit' 或 'lcs'。
    随机返回与用户提问最匹配的问题所对应的一个答案（代码或文本）。
    """
    key = best_match_by_lcs(question) if method == 'lcs' else best_match_by_edit(question)
    if not key:
        return "抱歉，我不清楚如何回答。"
    answers = qa_dict.get(key, [])
    return random.choice(answers) if answers else "抱歉，我不清楚如何回答。"

def main():
    print("欢迎使用智能问答，输入“退出”或“exit”结束对话。")
    method = 'lcs'  # 默认匹配方法，可根据需要改为 'lcs'
    while True:
        question = input("\n请输入你的问题：").strip()
        if not question:
            continue
        # 支持中英文退出
        if question.lower() in ('退出', 'exit', 'quit'):
            print("再见！")
            break
        # 获取回答并打印
        answer = answer_question(question, method=method)
        print(f"回答：\n{answer}")


main()
