docs.json
"""
[
  {
    "title": "神经系统总体结构与功能",
    "contents": [
      "神经系统由中枢神经系统（包括脑和脊髓）与周围神经系统（包括脑神经和脊神经）两部分组成，共同承担感知、传导、整合和控制等多种功能。",
      "中枢神经系统通过感受器接收外界和体内各种刺激，将信号在神经元网络中加工整合，形成运动指令、调节内脏活动或产生高级认知过程，如思维、记忆与情感。",
      "周围神经系统分为躯体神经和自主神经两大类：躯体神经主要控制骨骼肌的运动与皮肤、肌腱等部位的感觉；自主神经系统（包括交感神经与副交感神经）则调节心血管、消化、呼吸、泌尿以及内分泌等内脏功能，维持体内稳态。"
    ]
  },
  {
    "title": "神经元结构与信号传导机制",
    "contents": [
      "神经元是神经系统的基本结构和功能单位，可分为细胞体、树突和轴突三部分：细胞体内含有细胞核和大部分代谢结构，树突负责接收突触输入，轴突用于将动作电位传输出去并在末端形成突触。",
      "当局部膜电位达到阈值后，电压门控钠离子通道迅速开放，引发快速去极化；随后钾离子通道开放，钠通道失活，细胞膜重新极化并进入超极化阶段，完成一次动作电位全过程。",
      "在轴突末梢，动作电位触发钙离子通道开放，Ca²⁺内流促进突触囊泡与细胞膜融合，释放神经递质（如谷氨酸、GABA、乙酰胆碱等），这些化学信号跨越突触间隙，与下游神经元或效应细胞的受体结合，诱发新的电位变化或细胞反应。"
    ]
  },
  {
    "title": "神经可塑性与学习记忆",
    "contents": [
      "神经可塑性是指神经系统在发育、学习、伤后修复等过程中，通过突触强度变化、神经元再生及网络重组来调整自身结构和功能的能力。",
      "短期可塑性包括突触后电位的瞬时增强或减弱，如突触前钙离子累积导致的突触后电位增强；长期可塑性则以长期增强（LTP）和长期抑制（LTD）为典型代表，分别对应突触连接强度的持久增加或减弱。",
      "在海马体等与学习记忆密切相关的脑区，LTP通常伴随AMPA受体数量的增加和NMDA受体活性的调节，从而加强特定神经回路；相反，LTD会降低受体密度或改变突触前递质释放概率，帮助清除冗余信息，保持神经网络的灵活性。"
    ]
  },
  {
    "title": "动物分类学与主要演化分支",
    "contents": [
      "动物界依据形态学、发育学及分子生物学证据可分为无脊椎动物门（如海绵动物、腔肠动物、环节动物、软体动物和节肢动物）与脊索动物门（含鱼类、两栖类、爬行类、鸟类和哺乳类）。",
      "无脊椎动物在形态和生态位上多样性极高，从滤食的海绵到捕食的章鱼均有分布；节肢动物（尤其昆虫）数量最为庞大，几乎占据所有陆地与淡水生态系统的主导地位。",
      "脊索动物门的共同特征是早期胚胎阶段具有中轴的脊索、背部神经管和咽部裂孔。随着演化，骨骼系统（软骨或骨质）和内膜胚层衍生出的复杂器官系统（心血管、肺呼吸或鳃呼吸）逐步形成，使这些动物能占据多种栖息地并表现出高度的行为和生理适应性。"
    ]
  },
  {
    "title": "动物行为生态与适应性策略",
    "contents": [
      "动物行为涵盖觅食、防御、社交、繁殖及迁徙等多方面，其统计规律和神经生理基础反映了个体对内外环境刺激的综合适应策略。",
      "捕食者与猎物之间的“军备竞赛”推动了伪装、警戒信号及群体防御等行为的进化；社会性动物（如蜜蜂、蚂蚁和某些哺乳动物）通过分工协作、信息交流和集体决策，实现高效资源利用和风险分担。",
      "季节性迁徙（如候鸟和鲑鱼）依赖于内置的生物钟、磁场感受以及视觉、气味线索的多重导航系统；在恶劣环境中休眠或夏眠则是通过代谢率降低、能量消耗最小化来度过资源匮乏期的关键生存策略。"
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
