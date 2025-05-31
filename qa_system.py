# -*- coding: utf-8 -*-
"""
Q&A System with Chinese Tokenization and Interactive Conversation (with Programming Support)

This module implements a question-answering system with multiple matching algorithms:
  - Edit Distance (using fuzzywuzzy's fuzz.ratio)
  - Longest Common Subsequence (LCS)
  - Cosine similarity based on TF-IDF vectorization

A combined strategy is available to merge scores from all methods:
  1. Normalize all similarity scores to the range [0, 100].
  2. Calculate the weighted sum.
  3. Return the answer of the candidate question with the highest composite score.

Chinese tokenization is performed by jieba.
Additionally, the system supports programming Q&A with properly formatted code block outputs.
"""

# ----------------------
# Import required libraries
# ----------------------
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# ----------------------
# Define a Chinese tokenizer function using jieba
# ----------------------
def chinese_tokenizer(text):
    """Tokenize Chinese text using jieba."""
    return list(jieba.cut(text))

# ----------------------
# Define the knowledge base: question-answer pairs
# ----------------------
qa_pairs = [
    {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
    {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
    {"question": "如何使用Python进行数据分析？", "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。"},
    {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
    {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"},
    # Programming related Q&A
    {"question": "如何编写Python函数？", "answer": "You can define a Python function using the 'def' keyword. For example:\n\n```python\ndef my_function(param1, param2):\n    '''This function does something.'''\n    result = param1 + param2\n    return result\n```"}
]

# ----------------------
# Extract all questions from the knowledge base
# ----------------------
questions = [pair["question"] for pair in qa_pairs]

# ----------------------
# Build the TF-IDF model with custom Chinese tokenization
# ----------------------
vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
tfidf_matrix = vectorizer.fit_transform(questions)

# ----------------------
# LCS (Longest Common Subsequence) Functions
# ----------------------
def compute_lcs_length(s1, s2):
    """
    Compute the length of the longest common subsequence (LCS) between two strings.
    
    Parameters:
        s1: First string.
        s2: Second string.
    
    Returns:
        Length of LCS.
    """
    n = len(s1)
    m = len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def lcs_ratio(s1, s2):
    """
    Calculate the similarity percentage based on LCS:
      ratio = (2 * LCS_length) / (len(s1) + len(s2)) * 100
      
    Parameters:
        s1: First string.
        s2: Second string.
        
    Returns:
        Similarity percentage in the range [0, 100].
    """
    if not s1 and not s2:
        return 100
    lcs_len = compute_lcs_length(s1, s2)
    ratio = (lcs_len * 2) / (len(s1) + len(s2)) * 100
    return ratio

# ----------------------
# TF-IDF Similarity Score Function
# ----------------------
def tfidf_score(user_question, candidate_question):
    """
    Calculate the cosine similarity between the TF-IDF vectors of the user question
    and a candidate question.
    
    Parameters:
        user_question: The user input question.
        candidate_question: A candidate question from the knowledge base.
        
    Returns:
        Cosine similarity multiplied by 100 (range [0, 100]).
    """
    user_vec = vectorizer.transform([user_question])
    candidate_vec = vectorizer.transform([candidate_question])
    cosine_score = (user_vec * candidate_vec.T).toarray()[0][0]
    return cosine_score * 100

# ----------------------
# Combined Matching Strategy using All Methods
# ----------------------
def search_by_all_methods(user_question, threshold=50):
    """
    Combine Edit Distance, LCS, and TF-IDF to compute a weighted total similarity score,
    and return the index of the best candidate from the knowledge base.
    
    Parameters:
        user_question: The user input question.
        threshold: Minimum threshold to consider a candidate (range: 0-100).
        
    Returns:
        Index of the best matching question.
    """
    # Weight settings for each method
    weight_edit = 0.4    # Weight for Edit Distance
    weight_lcs = 0.3     # Weight for LCS
    weight_tfidf = 0.3   # Weight for TF-IDF

    best_idx = -1
    best_total_score = -1

    for i, candidate in enumerate(questions):
        score_edit = fuzz.ratio(user_question, candidate)  # Edit Distance score
        score_lcs = lcs_ratio(user_question, candidate)      # LCS similarity score
        score_tfidf = tfidf_score(user_question, candidate)  # TF-IDF cosine score

        total_score = weight_edit * score_edit + weight_lcs * score_lcs + weight_tfidf * score_tfidf

        if total_score >= threshold and total_score > best_total_score:
            best_total_score = total_score
            best_idx = i

    # If no candidate meets the threshold, choose the one with the highest score.
    if best_idx == -1:
        for i, candidate in enumerate(questions):
            score_edit = fuzz.ratio(user_question, candidate)
            score_lcs = lcs_ratio(user_question, candidate)
            score_tfidf = tfidf_score(user_question, candidate)
            total_score = weight_edit * score_edit + weight_lcs * score_lcs + weight_tfidf * score_tfidf
            if total_score > best_total_score:
                best_total_score = total_score
                best_idx = i

    return best_idx

# ----------------------
# Function to Retrieve the Best Answer
# ----------------------
def get_best_answer(user_question, method="tfidf", threshold=50):
    """
    Retrieve the best answer from the knowledge base by matching the user question
    using the specified algorithm.
    
    Supported methods:
      - "edit_distance": Use only Edit Distance for matching.
      - "lcs": Use only Longest Common Subsequence for matching.
      - "tfidf": Use only TF-IDF cosine similarity for matching.
      - "all": Use a combined strategy merging all three methods.
    
    Parameters:
        user_question: The user input question.
        method: The matching algorithm to use (default is "tfidf").
        threshold: Similarity threshold for single or combined strategies (range: 0-100).
        
    Returns:
        The best matching answer as a string.
    """
    if method == "edit_distance":
        best_idx = -1
        best_score = 0
        for i, candidate in enumerate(questions):
            score = fuzz.ratio(user_question, candidate)
            if score >= threshold and score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            for i, candidate in enumerate(questions):
                score = fuzz.ratio(user_question, candidate)
                if score > best_score:
                    best_score = score
                    best_idx = i

    elif method == "lcs":
        best_idx = -1
        best_score = 0
        for i, candidate in enumerate(questions):
            score = lcs_ratio(user_question, candidate)
            if score >= threshold and score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            for i, candidate in enumerate(questions):
                score = lcs_ratio(user_question, candidate)
                if score > best_score:
                    best_score = score
                    best_idx = i

    elif method == "tfidf":
        best_idx = -1
        best_score = -1
        for i, candidate in enumerate(questions):
            score = tfidf_score(user_question, candidate)
            if score > best_score:
                best_score = score
                best_idx = i

    elif method == "all":
        best_idx = search_by_all_methods(user_question, threshold=threshold)

    else:
        raise ValueError("Unsupported algorithm. Please choose: edit_distance, lcs, tfidf, or all")

    return qa_pairs[best_idx]["answer"]

# ----------------------
# Interactive Conversation Loop with Programming Support
# ----------------------
print("Welcome to the Q&A System!")
print("Please choose the matching algorithm:")
print("  1 - Edit Distance (edit_distance)")
print("  2 - Longest Common Subsequence (lcs)")
print("  3 - TF-IDF + Cosine Similarity (tfidf)")
print("  4 - Combined Matching (all)")

mode_input = input("Enter the matching algorithm (number or name): ").strip()
if mode_input in ["1", "edit_distance"]:
    method = "edit_distance"
elif mode_input in ["2", "lcs"]:
    method = "lcs"
elif mode_input in ["3", "tfidf"]:
    method = "tfidf"
elif mode_input in ["4", "all"]:
    method = "all"
else:
    print("Invalid input. Defaulting to tfidf mode.")
    method = "tfidf"

print("Selected matching mode:", method)
print("Enter your question (type 'quit', 'exit', or '退出' to quit).")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["quit", "exit", "退出"]:
        print("Thank you for using the system. Goodbye!")
        break

    # If the question contains common programming keywords or code block markers, 
    # it will support programming Q&A.
    if "python" in user_input.lower() or "def " in user_input or "代码" in user_input or "编程" in user_input or "```" in user_input:
        # Using get_best_answer will also match the programming Q&A if available.
        answer = get_best_answer(user_input, method=method, threshold=50)
    else:
        answer = get_best_answer(user_input, method=method, threshold=50)

    print("System:", answer)
