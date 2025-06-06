docs.json
"""
[
  {
    "title": "Python 基础",
    "contents": [
      "Python 是一种高级编程语言，语法简洁易读。",
      "它支持面向对象、过程化及函数式编程范式。",
      "拥有丰富的标准库和第三方包，适合 Web、数据分析、自动化运维等场景。"
    ]
  },
  {
    "title": "机器学习入门",
    "contents": [
      "机器学习是一种通过数据训练模型使其具有预测或决策能力的技术。",
      "常见算法包括线性回归、决策树、支持向量机、神经网络等。",
      "广泛应用于图像识别、自然语言处理、推荐系统等领域。"
    ]
  },
  {
    "title": "前端开发技术栈",
    "contents": [
      "前端开发主要使用 HTML、CSS、JavaScript。",
      "主流框架有 React、Vue、Angular 等。",
      "还会用到构建工具（Webpack、Rollup）和打包发布流程。"
    ]
  }
]
"""





# -*- coding: utf-8 -*-
"""
ir_lcs_edit.py

基于 LCS（最长公共子序列）和编辑距离的简单全文检索示例，
支持同一个 title 下有多个 content，并且可按 pick_mode
（one/few/all）灵活决定每个 title 实际参与检索的 content。
"""
import json
import random
# ------------------------------------------------------------------------------
# 1. LCS（Longest Common Subsequence）相关
# ------------------------------------------------------------------------------
def lcs_length(a: str, b: str) -> int:
    """
    计算字符串 a 和 b 的最长公共子序列长度（动态规划实现）。
    时间复杂度 O(len(a)*len(b))。
    """
    n, m = len(a), len(b)
    # dp[i][j] 表示 a[:i] 与 b[:j] 的 LCS 长度
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                # 取左边或上边的最大值
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]
def normalize_lcs(query: str, text: str) -> float:
    """
    将 LCS 长度归一化到 [0,1]，分子是 LCS 长度，分母是两字符串最大长度。
    """
    if not query or not text:
        return 0.0
    lcs_len = lcs_length(query, text)
    max_len = max(len(query), len(text))
    return lcs_len / max_len
# ------------------------------------------------------------------------------
# 2. 编辑距离（Levenshtein Distance）相关
# ------------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """
    计算字符串 a, b 的 Levenshtein 编辑距离（插入/删除/替换都算 1 步）。
    时间复杂度 O(len(a)*len(b))。
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # 边界初始化
    for i in range(n + 1):
        dp[i][0] = i   # a[:i] 变成 空串 需要 i 次删除
    for j in range(m + 1):
        dp[0][j] = j   # 空串 变成 b[:j] 需要 j 次插入

    # 状态转移
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除 a[i-1]
                dp[i][j - 1] + 1,      # 在 a 前插入 b[j-1]
                dp[i - 1][j - 1] + cost  # 替换 a[i-1] -> b[j-1]（如果相等则 cost=0）
            )
    return dp[n][m]
def normalize_edit_dist(query: str, text: str) -> float:
    """
    将编辑距离转换为相似度 [0,1]，公式 sim = 1 - dist / max_len。
    """
    # 两者都为空串，相似度定义为 1
    if not query and not text:
        return 1.0
    max_len = max(len(query), len(text))
    if max_len == 0:
        return 1.0
    dist = edit_distance(query, text)
    sim = 1.0 - dist / max_len
    return max(0.0, sim)
# ------------------------------------------------------------------------------
# 3. 综合打分：在 title 与 content 两部分分别计算 LCS 与 编辑距离，再加权合并
# ------------------------------------------------------------------------------
def score_doc(query: str,
              title: str,
              content: str,
              w_lcs: float, w_edit: float,
              w_title: float, w_content: float) -> float:
    """
    对单条“(title, content)”计算综合相似度得分：
     1) 标题相似度 = w_lcs * norm_lcs(query,title) + w_edit * norm_edit(query,title)
     2) 正文相似度 = w_lcs * norm_lcs(query,content) + w_edit * norm_edit(query,content)
     3) 最终得分     = w_title * 标题相似度 + w_content * 正文相似度
    """
    # 1. 标题部分
    lcs_t = normalize_lcs(query, title)
    edit_t = normalize_edit_dist(query, title)
    sim_title = w_lcs * lcs_t + w_edit * edit_t
    # 2. 正文部分
    lcs_c = normalize_lcs(query, content)
    edit_c = normalize_edit_dist(query, content)
    sim_content = w_lcs * lcs_c + w_edit * edit_c
    # 3. 加权合并
    return w_title * sim_title + w_content * sim_content
# ------------------------------------------------------------------------------
# 4. 加载文档：支持在同一个 title 下随机选“一条”/“几条”/“全部” contents
# ------------------------------------------------------------------------------
def load_docs(json_path: str, pick_mode: str = "one"):
    """
    从 JSON 文件加载文档，并根据 pick_mode 决定每个 title 对应的 content：
      - pick_mode="one" : 随机选取 1 条 content
      - pick_mode="few" : 随机选取最多 2 条 content
      - pick_mode="all" : 全部 contents 都加入
    返回：
      docs list，元素是 (title, content)，在后续 search() 中作为单条记录处理
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs = []
    for item in data:
        title = item.get('title', '')
        contents = item.get('contents', [])
        # 向后兼容：若 JSON 中使用旧的 "content" 字段，转换成列表
        if not isinstance(contents, list):
            single = item.get('content', '')
            contents = [single] if single else []
        # 根据 pick_mode 选取
        if pick_mode == "all":
            selected = contents
        elif pick_mode == "few":
            # 随机最多取 2 条
            k = 2
            if len(contents) <= k:
                selected = contents
            else:
                selected = random.sample(contents, k)
        else:  # 默认 "one"
            if contents:
                selected = [random.choice(contents)]
            else:
                selected = []
        # 将每个 (title, content) 组合作为独立文档加入列表
        for c in selected:
            docs.append((title, c))
    return docs
# ------------------------------------------------------------------------------
# 5. 检索函数 search：遍历 docs，对每条 (title,content) 打分，排序，取前 top_k
# ------------------------------------------------------------------------------
def search(query: str,
           docs: list,
           top_k: int,
           w_lcs: float, w_edit: float,
           w_title: float, w_content: float):
    """
    query   : 用户输入的查询字符串
    docs    : load_docs 返回的 [(title, content), ...]
    top_k   : 返回最相似的前 top_k 条
    w_lcs   : LCS 相似度在标题/正文内的权重
    w_edit  : 编辑距离相似度在标题/正文内的权重
    w_title : 标题相似度在最终得分中的乘权
    w_content: 正文相似度在最终得分中的乘权
    返回列表：[(title, content, score), ...]，按照 score 降序排列
    """
    results = []
    for title, content in docs:
        sc = score_doc(query,
                       title, content,
                       w_lcs, w_edit,
                       w_title, w_content)
        results.append((title, content, sc))
    # 按 score 降序
    results.sort(key=lambda x: x[2], reverse=True)
    # 取前 top_k
    return results[:top_k]
# ------------------------------------------------------------------------------
# 6. 主流程：保持原样，只在调用 load_docs 时传入 pick_mode
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 加载文档，示例中用 pick_mode="few"，你也可以选 "one" 或 "all"
    docs = load_docs("docs.json", pick_mode="few")
    # 2. 检索参数配置（可根据场景调节）
    w_lcs     = 0.5   # LCS 相似度内部权重
    w_edit    = 0.5   # 编辑距离相似度内部权重
    w_title   = 2.0   # 标题部分得分乘权
    w_content = 1.0   # 正文部分得分乘权
    top_k     = 3     # 返回前 top_k 条结果
    print("输入查询后按回车，输入 exit 或 quit 退出。")
    while True:
        query = input("\n查询> ").strip()
        if query.lower() in ("exit", "quit"):
            print("退出检索。")
            break
        if not query:
            print("查询不能为空，请重新输入。")
            continue
        # 3. 调用检索
        results = search(query, docs, top_k,
                         w_lcs, w_edit, w_title, w_content)
        # 4. 输出结果
        print(f"\nTop {len(results)} 结果：")
        for idx, (title, content, score) in enumerate(results, 1):
            print(f"{idx}. {title} (score={score:.4f})")
            print(f"   {content}")
