# -*- coding: utf-8 -*-
"""
Flask-based English QA system with multi-algorithm matching.
Supports multi-line, correctly indented code blocks directly stored as multi-line strings.
Frontend preserves indentation & line breaks via <pre> with monospace font.
User input textarea supports multiline; Ctrl+Enter to send.
Bootstrap4 responsive dark theme with red accent.
"""

from flask import Flask, request, jsonify, render_template_string
import jieba
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# ----------------------------
# English Q&A pairs with properly indented multi-line code blocks stored as triple-quoted strings
# No manual \n usage, direct multiline strings keep indentation naturally
# ----------------------------
qaPairs = [
    {
        "question": "What is Python?",
        "answer": """Python is a versatile programming language used for scripting,
automation, data analysis, web development, and more."""
    },
    {
        "question": "How do I write a for loop in Python?",
        "answer": """Here is a basic for loop example in Python:

```python
for i in range(5):
    print(i)
```"""
    },
    {
        "question": "What is a virtual environment?",
        "answer": """A virtual environment is an isolated environment that allows you to
manage dependencies for different Python projects separately.

To create one:

```bash
python -m venv env
source env/bin/activate  # On Linux/macOS
env\\Scripts\\activate   # On Windows
```"""
    },
    {
        "question": "Explain list comprehensions in Python.",
        "answer": """List comprehensions provide a concise way to create lists.

Example:

```python
squares = [x**2 for x in range(10)]
print(squares)
```"""
    },
    {
        "question": "How to handle exceptions in Python?",
        "answer": """Use try-except blocks to handle exceptions:

```python
try:
    risky_operation()
except Exception as e:
    print(f"Error occurred: {e}")
```"""
    }
]

# Extract questions list from QA pairs
questions = []
for pair in qaPairs:
    questions.append(pair["question"])

# ----------------------------
# Jieba tokenizer for both Chinese/English inputs (English tokenizes roughly as characters/words)
# ----------------------------
def jiebaTokenizer(text):
    tokens = jieba.cut(text)
    return list(tokens)

# ----------------------------
# Initialize and fit TF-IDF Vectorizer with jieba tokenizer
# ----------------------------
vectorizer = TfidfVectorizer(tokenizer=jiebaTokenizer)
tfidfMatrix = vectorizer.fit_transform(questions)

# ----------------------------
# Longest Common Subsequence (LCS) functions
# ----------------------------
def computeLcsLength(s1, s2):
    n = len(s1)
    m = len(s2)
    dp = []
    i = 0
    while i <= n:
        dp.append([0] * (m + 1))
        i += 1
    i = 0
    while i < n:
        j = 0
        while j < m:
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
            j += 1
        i += 1
    return dp[n][m]

def lcsRatio(s1, s2):
    if not s1 and not s2:
        return 100.0
    lcsLen = computeLcsLength(s1, s2)
    return (2.0 * lcsLen) / (len(s1) + len(s2)) * 100.0

# ----------------------------
# TF-IDF cosine similarity calculation
# ----------------------------
def tfidfScore(userQuestion, candidateQuestion):
    v1 = vectorizer.transform([userQuestion])
    v2 = vectorizer.transform([candidateQuestion])
    score = (v1 * v2.T).toarray()[0][0]
    return score * 100.0

# ----------------------------
# Search best matching answer with requested method
# ----------------------------
def searchBestAnswer(userQuestion, method="tfidf"):
    scoreList = []
    index = 0
    while index < len(questions):
        candidate = questions[index]

        if method == "editDistance":
            score = fuzz.ratio(userQuestion, candidate)
        elif method == "lcs":
            score = lcsRatio(userQuestion, candidate)
        elif method == "tfidf":
            score = tfidfScore(userQuestion, candidate)
        elif method == "fusion":
            edScore = fuzz.ratio(userQuestion, candidate)
            lcsScore = lcsRatio(userQuestion, candidate)
            tfidfSc = tfidfScore(userQuestion, candidate)
            score = edScore * 0.4 + lcsScore * 0.3 + tfidfSc * 0.3
        else:
            score = tfidfScore(userQuestion, candidate)

        scoreList.append((index, score))
        index += 1

    scoreList.sort(key=lambda x: x[1], reverse=True)
    bestIndex, bestScore = scoreList[0]
    answer = qaPairs[bestIndex]["answer"]
    return answer, round(bestScore, 2)

# ----------------------------
# Serve main page with bootstrap, dark theme and red font, multiline input, Ctrl+Enter to send
# front-end preserves indentation and line breaks via <pre style="white-space: pre-wrap;">
# ----------------------------
@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <title>QA System - Dark Mode</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <style>
    body,html {
      height: 100%;
      background-color: #121212 !important;
      color: #FF4444 !important;
      font-weight: 500;
    }
    #chatContainer {
      background-color: #1E1E1E;
      border: 1px solid #FF4444;
      border-radius: 0.5rem;
      padding: 1rem;
      height: 70vh;
      overflow-y: auto;
      margin-bottom: 1rem;
      max-width: 700px;
      margin-left: auto;
      margin-right: auto;
    }
    .chatBubble {
      max-width: 75%;
      padding: 0.75rem 1rem;
      border-radius: 1rem;
      margin-bottom: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 1rem;
      line-height: 1.4;
      box-shadow: 0 2px 5px rgb(255 68 68 / 0.3);
      font-family: Consolas, "Courier New", monospace;
    }
    .userMsg {
      background-color: #330000;
      color: #FF6666;
      margin-left: auto;
      text-align: right;
    }
    .systemMsg {
      background-color: #440000;
      color: #FF9999;
      margin-right: auto;
      text-align: left;
    }
    #inputArea {
      max-width: 700px;
      margin: auto;
      margin-bottom: 2rem;
      display: flex;
      padding: 0;
    }
    #userInput {
      flex-grow: 1;
      font-size: 1rem;
      border-radius: 0.375rem;
      border: 1.5px solid #FF4444;
      background-color: #121212;
      color: #FF6666;
      padding-left: 0.75rem;
      padding-right: 0.75rem;
      resize: vertical;
      min-height: 40px;
      max-height: 100px;
      outline: none;
    }
    #sendButton {
      margin-left: 0.75rem;
      background-color: #FF4444;
      color: #121212;
      border: none;
      border-radius: 0.375rem;
      font-weight: 600;
      width: 100px;
    }
    #sendButton:hover {
      background-color: #FF6666;
      color: #121212;
    }
    #algoSelect {
      margin-left: 0.75rem;
      max-width: 180px;
      border-radius: 0.375rem;
      border: 1.5px solid #FF4444;
      background-color: #121212;
      color: #FF6666;
      font-weight: 500;
      padding: 0.375rem 0.75rem;
      font-size: 1rem;
    }
    #chatContainer::-webkit-scrollbar {
      width: 8px;
    }
    #chatContainer::-webkit-scrollbar-thumb {
      background-color: #FF4444;
      border-radius: 4px;
    }
    #chatContainer::-webkit-scrollbar-track {
      background-color: #1E1E1E;
    }
    @media (max-width: 576px) {
      #chatContainer {
        height: 60vh;
        max-width: 95vw;
      }
      #inputArea {
        flex-direction: column;
      }
      #sendButton, #algoSelect {
        width: 100%;
        margin: 0.5rem 0 0 0;
      }
      #algoSelect {
        margin-left: 0;
      }
      #userInput {
        margin-bottom: 0.5rem;
      }
    }
  </style>
</head>
<body>
  <div id="chatContainer" aria-live="polite" aria-atomic="false" aria-relevant="additions" role="log"></div>

  <div id="inputArea" class="container d-flex justify-content-center align-items-center">
    <textarea id="userInput" placeholder="Enter your question, supports multiline, press Ctrl+Enter to send"
      aria-label="Enter your question" autocomplete="off" rows="2"></textarea>
    <select id="algoSelect" aria-label="Select matching algorithm">
      <option value="tfidf">TF-IDF Cosine Similarity</option>
      <option value="editDistance">Edit Distance</option>
      <option value="lcs">Longest Common Subsequence</option>
      <option value="fusion">Weighted Fusion (Recommended)</option>
    </select>
    <button id="sendButton" aria-label="Send question">Send</button>
  </div>

  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script>
    // Append message, use <pre> to preserve formatting
    function appendMessage(msg, sender, similarity){
      let bubbleDiv = document.createElement("div");
      bubbleDiv.classList.add("chatBubble");
      bubbleDiv.classList.add(sender === "user" ? "userMsg" : "systemMsg");

      let pre = document.createElement("pre");
      pre.style.whiteSpace = "pre-wrap";
      pre.textContent = msg; // safe & preserves indentation & line breaks

      bubbleDiv.appendChild(pre);

      if(sender === "system" && similarity !== undefined){
        let simDiv = document.createElement("div");
        simDiv.style.fontSize = "0.85rem";
        simDiv.style.marginTop = "4px";
        simDiv.style.color = "#FFAAAA";
        simDiv.textContent = `(Similarity: ${similarity}%)`;
        bubbleDiv.appendChild(simDiv);
      }

      let chatContainer = document.getElementById("chatContainer");
      chatContainer.appendChild(bubbleDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Send user question via AJAX POST to /ask
    function sendQuestion(){
      let inputBox = document.getElementById("userInput");
      let question = inputBox.value.trim();
      if(question.length === 0) return; 

      appendMessage(question, "user");
      inputBox.value = "";
      inputBox.focus();

      let selectedAlgo = $("#algoSelect").val();

      $.ajax({
        url: "/ask",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({question: question, method: selectedAlgo}),
        success: function(res){
          if(res.answer){
            appendMessage(res.answer, "system", res.similarity);
          } else if(res.error){
            appendMessage("Error: " + res.error, "system");
          } else {
            appendMessage("No suitable answer found.", "system");
          }
        },
        error: function(){
          appendMessage("Request failed. Please try again later.", "system");
        }
      });
    }

    $(document).ready(function(){
      $("#sendButton").on("click", sendQuestion);
      // Ctrl+Enter sends, Enter inserts newline
      $("#userInput").on("keydown", function(event){
        if(event.key === "Enter" && event.ctrlKey){
          event.preventDefault();
          sendQuestion();
        }
      });
    });
  </script>
</body>
</html>
"""
    return render_template_string(html)

# ----------------------------
# /ask API: JSON POST returns answer and similarity score
# ----------------------------
@app.route("/ask", methods=["POST"])
def apiAsk():
    reqData = request.get_json()
    if not reqData:
        return jsonify({"error": "Empty request body"}), 400

    userQuestion = reqData.get("question", "").strip()
    algoMethod = reqData.get("method", "tfidf").strip()

    if userQuestion == "":
        return jsonify({"error": "Question is required"}), 400

    allowedMethods = {"editDistance", "lcs", "tfidf", "fusion"}
    if algoMethod not in allowedMethods:
        algoMethod = "tfidf"

    answer, similarity = searchBestAnswer(userQuestion, method=algoMethod)
    return jsonify({"answer": answer, "similarity": similarity})

# ----------------------------
# Run Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
