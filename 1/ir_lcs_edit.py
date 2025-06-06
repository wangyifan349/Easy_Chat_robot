# -*- coding: utf-8 -*-
import json

# 动态规划求最长公共子序列长度
def lcs_length(a, b):
    n = len(a)
    m = len(b)
    dp = []
    for i in range(n+1):
        row = []
        for j in range(m+1):
            row.append(0)
        dp.append(row)
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                if dp[i][j+1] > dp[i+1][j]:
                    dp[i+1][j+1] = dp[i][j+1]
                else:
                    dp[i+1][j+1] = dp[i+1][j]
    return dp[n][m]

# 动态规划求编辑距离（Levenshtein）
def edit_distance(a, b):
    n = len(a)
    m = len(b)
    dp = []
    for i in range(n+1):
        row = []
        for j in range(m+1):
            row.append(0)
        dp.append(row)
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1
            delete = dp[i-1][j] + 1
            insert = dp[i][j-1] + 1
            replace = dp[i-1][j-1] + cost
            # 取三者最小
            min_val = delete
            if insert < min_val:
                min_val = insert
            if replace < min_val:
                min_val = replace
            dp[i][j] = min_val
    return dp[n][m]

# 把 LCS 长度归一化到 [0,1]
def normalize_lcs(query, text):
    if len(query) == 0 or len(text) == 0:
        return 0.0
    lcs_len = lcs_length(query, text)
    max_len = len(query)
    if len(text) > max_len:
        max_len = len(text)
    return float(lcs_len) / float(max_len)

# 把编辑距离转成相似度 [0,1]
def normalize_edit_dist(query, text):
    if len(query) == 0 and len(text) == 0:
        return 1.0
    max_len = len(query)
    if len(text) > max_len:
        max_len = len(text)
    if max_len == 0:
        return 1.0
    dist = edit_distance(query, text)
    sim = 1.0 - float(dist) / float(max_len)
    if sim < 0.0:
        sim = 0.0
    return sim

# 对单条文档计算综合得分
def score_doc(query, title, content,
              w_lcs, w_edit, w_title, w_content):
    # 计算标题相似度
    lcs_t = normalize_lcs(query, title)
    edit_t = normalize_edit_dist(query, title)
    sim_title = w_lcs * lcs_t + w_edit * edit_t
    # 计算正文相似度
    lcs_c = normalize_lcs(query, content)
    edit_c = normalize_edit_dist(query, content)
    sim_content = w_lcs * lcs_c + w_edit * edit_c
    # 加权合并
    return w_title * sim_title + w_content * sim_content

# 从 JSON 文件加载文档列表
def load_docs(json_path):
    f = open(json_path, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    docs = []
    for item in data:
        title = item.get('title', '')
        content = item.get('content', '')
        docs.append((title, content))
    return docs

# 检索函数，返回 top_k 条 (title, content, score)
def search(query, docs, top_k,
           w_lcs, w_edit, w_title, w_content):
    results = []
    for doc in docs:
        title = doc[0]
        content = doc[1]
        score = score_doc(query,
                          title, content,
                          w_lcs, w_edit,
                          w_title, w_content)
        results.append((title, content, score))
    # 按 score 降序排序
    results.sort(key=lambda x: x[2], reverse=True)
    # 取前 top_k
    top_results = []
    count = 0
    for item in results:
        if count >= top_k:
            break
        top_results.append(item)
        count += 1
    return top_results
