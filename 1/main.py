from ir_lcs_edit import load_docs, search

# 1. 加载外部 JSON 文档
docs = load_docs("docs.json")

# 2. 定义查询和权重
query = "深度学习 框架 教程"
w_lcs = 0.5
w_edit = 0.5
w_title = 2.0
w_content = 1.0
top_k = 3

# 3. 执行检索
results = search(query, docs, top_k,
                 w_lcs, w_edit,
                 w_title, w_content)

# 4. 输出结果
print("Query:", query)
for idx in range(len(results)):
    title, content, score = results[idx]
    print(f"{idx+1}. {title} (score={score:.4f})")
    print("   ", content)
