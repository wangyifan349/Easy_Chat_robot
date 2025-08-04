# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

try:
    from rapidfuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz

jieba.initialize()

def chinese_tokenizer(text):
    return jieba.lcut(text)

qa_pairs = [
    {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
    {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
    {"question": "如何使用Python进行数据分析？", "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。"},
    {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
    {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"},
    {"question": "如何编写Python函数？", "answer": 
"""你可以使用 `def` 关键词来定义一个 Python 函数。例如：

```python
def my_function(param1, param2):
    '''这个函数执行某些操作。'''
    result = param1 + param2
    return result
```
"""}
]

questions = [pair["question"] for pair in qa_pairs]

vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
vectorizer.fit(questions)

def compute_lcs_length(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def lcs_ratio(s1, s2):
    if not s1 and not s2:
        return 100.0
    lcs_len = compute_lcs_length(s1, s2)
    return (2 * lcs_len) / (len(s1) + len(s2)) * 100

def tfidf_score(uq, cq):
    try:
        u_vec = vectorizer.transform([uq])
        c_vec = vectorizer.transform([cq])
        return (u_vec * c_vec.T).toarray()[0][0] * 100
    except Exception:
        return 0.0

def search_by_all_methods(uq, threshold=50, top_k=1):
    w_edit, w_lcs, w_tfidf = 0.4, 0.3, 0.3
    scored = []
    for idx, candidate in enumerate(questions):
        se = fuzz.ratio(uq, candidate)
        sl = lcs_ratio(uq, candidate)
        st = tfidf_score(uq, candidate)
        total = w_edit * se + w_lcs * sl + w_tfidf * st
        if total >= threshold:
            scored.append((idx, total))
    if not scored:
        for idx, candidate in enumerate(questions):
            se = fuzz.ratio(uq, candidate)
            sl = lcs_ratio(uq, candidate)
            st = tfidf_score(uq, candidate)
            total = w_edit * se + w_lcs * sl + w_tfidf * st
            scored.append((idx, total))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]

def get_best_answers(user_question, method="tfidf", threshold=50, top_k=1):
    if method == "edit_distance":
        scores = [(i, fuzz.ratio(user_question, q)) for i, q in enumerate(questions)]
        scores.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in scores[:top_k]]
    elif method == "lcs":
        scores = [(i, lcs_ratio(user_question, q)) for i, q in enumerate(questions)]
        scores.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in scores[:top_k]]
    elif method == "tfidf":
        scores = [(i, tfidf_score(user_question, q)) for i, q in enumerate(questions)]
        scores.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in scores[:top_k]]
    elif method == "all":
        indices = search_by_all_methods(user_question, threshold, top_k)
    else:
        raise ValueError("不支持的算法，请选择：edit_distance、lcs、tfidf 或 all。")
    return [qa_pairs[i]["answer"] for i in indices]

def main():
    print("欢迎使用问答系统！")
    print("可选匹配算法：1-edit_distance  2-lcs  3-tfidf  4-all")
    choice = input("请输入算法编号或名称：").strip()
    methods = {"1":"edit_distance","2":"lcs","3":"tfidf","4":"all",
               "edit_distance":"edit_distance","lcs":"lcs","tfidf":"tfidf","all":"all"}
    method = methods.get(choice, "tfidf")
    k_input = input("请输入返回答案数量 top k：").strip()
    top_k = int(k_input) if k_input.isdigit() and int(k_input)>0 else 1
    print(f"模式：{method}，top_k={top_k}。输入 '退出' 结束。")
    while True:
        user_q = input("用户：").strip()
        if user_q in ("退出","exit","quit"):
            print("再见！")
            break
        answers = get_best_answers(user_q, method=method, threshold=50, top_k=top_k)
        print("系统：")
        for ans in answers:
            print(ans)
        print("-"*40)

if __name__ == "__main__":
    main()
```
