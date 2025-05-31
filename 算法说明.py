import numpy as np
from collections import Counter
import math

# -------------------------
# 计算文本相似性的函数实现
# -------------------------
# 1. 最长公共子序列 (LCS)
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]  # 返回公共子序列的长度

# 2. 编辑距离 (Edit Distance)
def edit_distance(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if X[i-1] == Y[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

# 3. TF-IDF 和余弦相似度
def tf_idf(documents):
    # 构建词汇表（以空格分词，区分大小写，可根据需要扩展预处理方式）
    vocabulary = set()
    for document in documents:
        vocabulary.update(document.split())
    vocabulary = sorted(vocabulary)
    
    # 统计每个词在多少个文档中出现（文档频率）
    N = len(documents)
    doc_freq = {}
    for word in vocabulary:
        doc_freq[word] = sum(1 for doc in documents if word in doc.split())
    
    tf_idf_values = []
    for document in documents:
        words = document.split()
        word_count = len(words)
        counter = Counter(words)
        entry = {}
        for word in vocabulary:
            tf = counter[word] / word_count if word_count > 0 else 0
            idf = np.log(N / (doc_freq[word] if doc_freq[word] > 0 else 1))
            entry[word] = tf * idf
        tf_idf_values.append(entry)
    return tf_idf_values, vocabulary

def cosine_similarity(X, Y):
    dot_product = np.dot(X, Y)
    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    if norm_X == 0 or norm_Y == 0:
        return 0.0
    return dot_product / (norm_X * norm_Y)

def tf_idf_cosine_similarity(query, text):
    """
    将 query 与 text 两个文本看作文档，计算它们的 TF-IDF 余弦相似度。
    """
    documents = [query, text]
    tf_idf_values, vocabulary = tf_idf(documents)
    vector_query = np.array([tf_idf_values[0][word] for word in vocabulary])
    vector_text  = np.array([tf_idf_values[1][word] for word in vocabulary])
    return cosine_similarity(vector_query, vector_text)

# -------------------------
# 问答知识库与匹配函数
# -------------------------

# 预定义问答知识库，示例中增添了一个包含代码的问答对
qa_knowledge_base = [
    {
        "question": "今天的天气怎么样？",
        "answer": "今天天气晴朗，适合外出。"
    },
    {
        "question": "如何学习编程？",
        "answer": "建议多实践，多写代码，同时参考相关书籍和在线教程。"
    },
    {
        "question": "数据科学是什么？",
        "answer": "数据科学是利用统计、机器学习等方法从数据中获得知识和洞察。"
    },
    {
        "question": "什么是人工智能？",
        "answer": "人工智能是研究如何让计算机执行复杂任务（如视觉、语言理解、决策等）的科学。"
    },
    {
        "question": "如何在 Python 中实现冒泡排序？",
        "answer": (
            "下面是一个冒泡排序的示例代码：\n"
            "-------------------------------------------------\n"
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        # 提前退出标志\n"
            "        swapped = False\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "                swapped = True\n"
            "        if not swapped:\n"
            "            break\n"
            "    return arr\n"
            "-------------------------------------------------\n"
            "可以调用 bubble_sort([64, 34, 25, 12, 22, 11, 90]) 来测试。"
        ),
        "is_code": True
    }
]

def match_question(user_question, method="tfidf"):
    """
    根据选择的方法（lcs, edit, tfidf）在知识库中寻找与用户问题最相似的问题，并返回对应的问答对和匹配得分。
    """
    best_score = None
    best_match = None
    for qa in qa_knowledge_base:
        kb_question = qa["question"]
        if method == "lcs":
            # LCS 得分：公共子序列越长得分越高
            score = lcs(user_question, kb_question)
        elif method == "edit":
            # 编辑距离转化得分：距离越小得分越高（1/(distance+1)）
            distance = edit_distance(user_question, kb_question)
            score = 1 / (distance + 1)
        else:  # 默认 tfidf
            score = tf_idf_cosine_similarity(user_question, kb_question)
        if best_score is None or score > best_score:
            best_score = score
            best_match = qa
    return best_match, best_score
# -------------------------
# 持续问答模式
# -------------------------
def qa_loop():
    print("欢迎使用问答系统！")
    print("请选择相似度匹配方式：")
    print("1. 最长公共子序列 (输入 lcs)")
    print("2. 编辑距离 (输入 edit)")
    print("3. TF-IDF 余弦相似度 (输入 tfidf)")
    
    method_input = input("请输入匹配方式 (lcs/edit/tfidf) [默认 tfidf]: ").strip().lower()
    if method_input not in ["lcs", "edit", "tfidf"]:
        method = "tfidf"
    else:
        method = method_input
    
    print("\n问答系统开启（输入 'exit' 退出）")
    while True:
        user_question = input("\n请输入您的问题：").strip()
        if user_question.lower() == "exit":
            print("退出问答系统，再见！")
            break
        
        best_match, score = match_question(user_question, method=method)
        
        print("\n匹配到的问题：")
        print(best_match["question"])
        print("匹配得分：", score)
        
        print("\n回答：")
        # 检查是否为代码，如果是则以代码块格式输出
        if best_match.get("is_code"):
            print("─────────────────────────────")
            print(best_match["answer"])
            print("─────────────────────────────")
        else:
            print(best_match["answer"])

if __name__ == "__main__":
    qa_loop()
