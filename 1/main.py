from ir_lcs_edit import load_docs, search
# 1. 预先加载文档
docs = load_docs("docs.json")
# 2. 定义检索参数（可根据实际场景调整权重与结果条数）
w_lcs     = 0.5   # LCS（最长公共子序列）相似度权重
w_edit    = 0.5   # 编辑距离相似度权重
w_title   = 2.0   # 标题相似度在最终得分中的乘法权重
w_content = 1.0   # 正文相似度在最终得分中的乘法权重
top_k     = 2     # 每次检索返回的最相似结果条数
print("输入查询后按回车，输入 exit 或 quit 退出。")
while True:
    query = input("\n查询> ").strip()
    if query.lower() in ("exit", "quit"):
        print("退出检索。")
        break
    if query == "":
        print("查询不能为空，请重新输入。")
        continue
    results = search(query, docs, top_k,
                     w_lcs, w_edit, w_title, w_content)
    print(f"\nTop {len(results)} 结果:")
    idx = 1
    for title, content, score in results:
        print(f"{idx}. {title} (score={score:.4f})")
        print(f"   {content}")
        idx += 1
