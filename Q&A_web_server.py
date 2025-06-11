# -*- coding: utf-8 -*-
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
    # Programming
    {
        "question": "What is Python?",
        "answer": """Python is a high-level, general-purpose programming language known for its clear and readable syntax. It supports object-oriented, procedural, and functional programming paradigms, and comes with a rich standard library plus a vast third-party ecosystem. Python is widely used for scripting, automation, data analysis, machine learning, web development, scientific computing, and more."""
    },
    {
        "question": "How do I write a for loop in Python?",
        "answer": """In Python, you use the for keyword together with the built-in range function to iterate over a sequence of integers. For example:
for i in range(5):
    print(i)
This code will print 0, 1, 2, 3, 4 in sequence. You can also loop over any iterable (like lists, tuples, strings, dictionaries, etc.) using the same syntax."""
    },
    {
        "question": "What is a virtual environment?",
        "answer": """A virtual environment is an isolated Python runtime environment that keeps dependencies required by different projects separate, preventing version conflicts. The typical workflow is:
1. Create a new env folder: python -m venv env  
2. Activate it:  
   - On Linux/macOS: source env/bin/activate  
   - On Windows: env\\Scripts\\activate  
3. Install packages inside that env: pip install requests  
4. When done, deactivate with: deactivate"""
    },
    {
        "question": "Explain list comprehensions in Python.",
        "answer": """A list comprehension provides a concise way to create lists by embedding loops and optional conditionals in a single line. For example, to get the squares of numbers 0–9:
squares = [x * x for x in range(10)]
To include only even numbers:
even_squares = [x * x for x in range(10) if x % 2 == 0]"""
    },
    {
        "question": "What is a Python decorator?",
        "answer": """A decorator is a higher-order function that takes another function as an argument and returns a new function, allowing you to add functionality to the original function without modifying its code. You apply a decorator with the @ syntax:
def debug(fn):
    def wrapped(*args, **kwargs):
        print("Calling", fn.__name__)
        return fn(*args, **kwargs)
    return wrapped

@debug
def greet(name):
    print("Hello", name)

Calling greet("Alice") will first print “Calling greet” and then “Hello Alice”."""
    },
    {
        "question": "How do I read a file in Python?",
        "answer": """You typically use the built-in open function along with a with statement for proper resource management. For example:
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)

The file is automatically closed when the with block ends. To read line by line:
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line, end="")"""
    },
    {
        "question": "What is Git and how to clone a repository?",
        "answer": """Git is a distributed version control system for tracking file changes and collaborating on code. To clone a remote repository locally, use:
git clone https://github.com/username/repo.git
This will create a folder named repo in your current directory and download all commits and files."""
    },

    # Data Science / Machine Learning
    {
        "question": "What is Pandas in Python?",
        "answer": """Pandas is a powerful data manipulation and analysis library that provides the DataFrame object for working with tabular data. Common operations:
import pandas as pd

# Read from CSV
df = pd.read_csv("data.csv")

# View first 5 rows
print(df.head())

# Filter rows where age > 30
df_filtered = df[df["age"] > 30]"""
    },
    {
        "question": "How to train a simple linear regression model with scikit-learn?",
        "answer": """Here’s the typical workflow with scikit-learn:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Instantiate and train
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE:", mse)
print("R2 score:", r2)"""
    },

    # Physics
    {
        "question": "What is Newton's first law?",
        "answer": """Newton’s First Law (the law of inertia) states that an object at rest stays at rest, and an object in uniform motion continues in uniform straight-line motion, unless acted upon by an external force. It implies that forces are needed to change motion, not to maintain it."""
    },
    {
        "question": "Define kinetic energy.",
        "answer": """Kinetic energy is the energy an object possesses due to its motion. It depends on the object’s mass and velocity, and is given by K = 1/2 * m * v^2, where m is mass and v is velocity. Because of the square relationship, small changes in speed create larger changes in energy."""
    },
    {
        "question": "What is Ohm's law?",
        "answer": """Ohm’s Law describes the relationship between voltage (V), current (I), and resistance (R) in a conductor:  
V = I * R  
where V is the voltage across the conductor, I is the current through it, and R is its resistance. This holds for many metallic conductors at constant temperature."""
    },

    # Biology
    {
        "question": "What is DNA?",
        "answer": """Deoxyribonucleic acid (DNA) is the hereditary molecule that carries genetic information in living organisms. It consists of two complementary strands forming a double helix, with each strand made up of nucleotides containing one of four bases: adenine (A), thymine (T), cytosine (C), and guanine (G). DNA’s main functions are to store, replicate, and transmit genetic information."""
    },
    {
        "question": "Explain photosynthesis.",
        "answer": """Photosynthesis is the process by which green plants, algae, and some bacteria convert carbon dioxide and water into organic compounds (usually glucose) using light energy, releasing oxygen as a byproduct. The overall reaction is:  
6 CO2 + 6 H2O + light energy → C6H12O6 + 6 O2  
It consists of light-dependent reactions and the Calvin cycle (light-independent reactions), providing the primary energy source and oxygen for ecosystems."""
    },
    {
        "question": "What are the main differences between prokaryotic and eukaryotic cells?",
        "answer": """Key differences include:  
1. Nucleus: Prokaryotes lack a membrane-bound nucleus; DNA is in the cytoplasm. Eukaryotes have a double-membrane nucleus.  
2. Organelles: Prokaryotes have no membrane-bound organelles (only ribosomes). Eukaryotes possess mitochondria, ER, Golgi, etc.  
3. Gene expression: In prokaryotes, transcription and translation occur simultaneously; in eukaryotes, transcription happens in the nucleus and mRNA is processed before being translated in the cytoplasm.  
4. Size: Prokaryotes are typically 1–10 μm in diameter; eukaryotes range from 10–100 μm."""
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
