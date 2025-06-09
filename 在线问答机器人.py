# -*- coding: utf-8 -*-
"""
# 导入所需的库和模块
from flask import Flask, request, jsonify, render_template_string
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# 创建 Flask 应用
app = Flask(__name__)
# 示例问答知识库（部分答案包含换行和缩进）
qa_pairs = [
    {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
    {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
    {"question": "如何使用Python进行数据分析？", 
     "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。\n例如：\n    import pandas as pd\n    import numpy as np"},
    {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
    {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"}
]

# 将知识库中的问题提取到一个列表中
questions = []
for pair in qa_pairs:
    questions.append(pair["question"])

# 使用 jieba 分词函数，对文本进行分词处理
def jieba_tokenizer(text):
    segments = jieba.cut(text)
    tokens = []
    for token in segments:
        tokens.append(token)
    return tokens

# 创建 TF-IDF 模型，并设置 tokenizer 使用 jieba_tokenizer
vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
tfidf_matrix = vectorizer.fit_transform(questions)

# 根据 TF-IDF 相似度搜索最佳匹配问题的索引
def search_by_tfidf(user_question):
    # 将用户输入的问题转换为向量
    user_vec = vectorizer.transform([user_question])
    # 计算用户问题与知识库中每个问题的相似度
    similarity_array = (tfidf_matrix * user_vec.T).toarray().ravel()
    best_index = 0  # 初始化最佳索引为 0
    best_score = similarity_array[0]  # 初始化最佳得分
    i = 1
    # 遍历所有相似度得分，找到最大的得分对应的索引
    while i < len(similarity_array):
        if similarity_array[i] > best_score:
            best_score = similarity_array[i]
            best_index = i
        i = i + 1
    return best_index

# 根据用户输入的问题返回最佳答案
def get_best_answer(user_question):
    best_index = search_by_tfidf(user_question)  # 根据 TF-IDF 搜索最佳匹配索引
    best_answer = qa_pairs[best_index]["answer"]  # 获取最佳匹配问题的答案
    return best_answer

# 定义处理 /ask 路由的 POST 请求
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()  # 从请求中获取 JSON 数据
    user_question = data.get("question", "")  # 获取用户问题
    if user_question == "":
        return jsonify({"error": "请输入问题"}), 400  # 若问题为空，返回错误信息
    best_answer = get_best_answer(user_question)  # 获取最佳答案
    return jsonify({"answer": best_answer})  # 以 JSON 格式返回答案

# 定义处理根路径 "/" 的 GET 请求，返回网页内容
@app.route("/")
def index():
    # 定义网页 HTML 内容，包含基本的聊天式界面
    page_html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>在线问答 - 聊天式界面</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style type="text/css">
        body {
          background-color: #f5f5f5;
        }
        .chat-container {
          max-width: 600px;
          margin: 50px auto;
          background-color: #ffffff;
          border: 1px solid #dddddd;
          border-radius: 5px;
          padding: 15px;
          height: 500px;
          overflow-y: auto;
        }
        .chat-bubble {
          padding: 10px 15px;
          border-radius: 15px;
          margin-bottom: 10px;
          max-width: 80%;
          clear: both;
          white-space: pre-wrap;
        }
        .user-msg {
          background-color: #dcf8c6;
          float: right;
          text-align: right;
        }
        .system-msg {
          background-color: #f1f0f0;
          float: left;
          text-align: left;
        }
        .input-area {
          position: fixed;
          bottom: 0;
          width: 100%;
          background: #ffffff;
          padding: 10px 0;
          border-top: 1px solid #dddddd;
        }
        .input-box {
          max-width: 600px;
          margin: 0 auto;
        }
      </style>
    </head>
    <body>
      <div class="chat-container" id="chatContainer">
      </div>
      <div class="input-area">
        <div class="input-box container">
          <div class="input-group">
            <input type="text" class="form-control" id="questionInput" placeholder="请输入问题">
            <div class="input-group-append">
              <button class="btn btn-primary" id="sendBtn" type="button">发送</button>
            </div>
          </div>
        </div>
      </div>
      <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
      <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
      <script type="text/javascript">
        // 在页面中添加消息的函数，根据消息类型添加不同样式
        function appendMessage(message, type) {
          var bubbleClass = "";
          if (type === "user") {
            bubbleClass = "user-msg";
          } else {
            bubbleClass = "system-msg";
          }
          var bubble = $("<div></div>");
          bubble.addClass("chat-bubble");
          bubble.addClass(bubbleClass);
          bubble.text(message);
          $("#chatContainer").append(bubble);
          $("#chatContainer").scrollTop($("#chatContainer")[0].scrollHeight);
        }
        // 文档加载完成后的事件绑定
        $(document).ready(function(){
          // 绑定发送按钮点击事件
          $("#sendBtn").click(function(){
            var question = $("#questionInput").val().trim();
            if (question === "") {
              return;
            }
            appendMessage(question, "user");  // 添加用户消息到聊天框
            $("#questionInput").val("");  // 清空输入框
            // 通过 Ajax 发送 POST 请求给服务端 /ask 接口
            $.ajax({
              url: "/ask",
              type: "POST",
              contentType: "application/json",
              data: JSON.stringify({question: question}),
              success: function(response) {
                appendMessage(response.answer, "system");  // 显示系统返回的答案
              },
              error: function(error) {
                appendMessage("请求出错，请稍后再试。", "system");  // 显示错误信息
                console.log("错误信息：", error);
              }
            });
          });
          // 绑定键盘回车事件，按回车发送消息
          $("#questionInput").keypress(function(event){
            if(event.which === 13) {
              $("#sendBtn").click();
              return false;
            }
          });
        });
      </script>
    </body>
    </html>
    """
    # 使用 Flask 的 render_template_string 渲染 HTML 页面
    return render_template_string(page_html)

# 主程序入口，启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


"""
# -*- coding: utf-8 -*-

# 导入所需的库和模块
from flask import Flask, request, jsonify, render_template_string
import jieba
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json
import os

# 创建 Flask 应用
app = Flask(__name__)

# 从JSON文件加载问答对
def load_qa_pairs():
    if os.path.exists('qa_pairs.json'):
        with open('qa_pairs.json', 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
    else:
        # 如果JSON文件不存在，使用默认问答对
        qa_pairs = [
            {"question": "今天天气怎么样？", "answer": "今天天气晴朗，适合外出。"},
            {"question": "你叫什么名字？", "answer": "我是一个智能问答系统。"},
            {"question": "如何使用Python进行数据分析？",
             "answer": "可以使用Pandas、NumPy、Matplotlib等库进行数据分析。\n例如：\n    import pandas as pd\n    import numpy as np"},
            {"question": "中国的首都是哪里？", "answer": "中国的首都是北京。"},
            {"question": "Python能做什么？", "answer": "Python具有丰富的生态系统，可用于数据分析、机器学习、网络爬虫等。"}
        ]
    return qa_pairs

qa_pairs = load_qa_pairs()

# 将知识库中的问题提取到一个列表中
questions = [pair["question"] for pair in qa_pairs]

# 加载预训练的中文Sentence-BERT模型
model = SentenceTransformer('uer/sbert-base-chinese-nli')

# 对知识库中的问题进行编码
question_embeddings = model.encode(questions)

# 定义函数计算相似度
def search_by_embedding(user_question):
    # 对用户的问题进行编码
    user_embedding = model.encode([user_question])[0]
    # 计算余弦相似度
    similarity_array = np.dot(question_embeddings, user_embedding) / (
        np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(user_embedding))
    # 找到最大相似度的索引
    best_index = np.argmax(similarity_array)
    best_score = similarity_array[best_index]
    return best_index, best_score

# 根据用户输入的问题返回最佳答案
def get_best_answer(user_question):
    best_index, best_score = search_by_embedding(user_question)
    if best_score < 0.5:  # 相似度阈值，可根据实际情况调整
        return "抱歉，我无法理解您的问题，请您尝试换种方式提问。"
    best_answer = qa_pairs[best_index]["answer"]
    return best_answer

# 定义处理 /ask 路由的 POST 请求
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()  # 从请求中获取 JSON 数据
    user_question = data.get("question", "")  # 获取用户问题
    if user_question == "":
        return jsonify({"error": "请输入问题"}), 400  # 若问题为空，返回错误信息
    best_answer = get_best_answer(user_question)  # 获取最佳答案
    return jsonify({"answer": best_answer})  # 以 JSON 格式返回答案

# 定义处理根路径 "/" 的 GET 请求，返回网页内容
@app.route("/")
def index():
    # 定义网页 HTML 内容，包含基本的聊天式界面
    page_html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>在线问答 - 聊天式界面</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style type="text/css">
        body {
          background-color: #f5f5f5;
        }
        .chat-container {
          max-width: 600px;
          margin: 50px auto;
          background-color: #ffffff;
          border: 1px solid #dddddd;
          border-radius: 5px;
          padding: 15px;
          height: 500px;
          overflow-y: auto;
        }
        .chat-bubble {
          padding: 10px 15px;
          border-radius: 15px;
          margin-bottom: 10px;
          max-width: 80%;
          clear: both;
          white-space: pre-wrap;
          position: relative;
        }
        .user-msg {
          background-color: #dcf8c6;
          float: right;
          text-align: right;
        }
        .system-msg {
          background-color: #f1f0f0;
          float: left;
          text-align: left;
        }
        .timestamp {
          font-size: 10px;
          color: #999;
          position: absolute;
          bottom: -15px;
          right: 15px;
        }
        .input-area {
          position: fixed;
          bottom: 0;
          width: 100%;
          background: #ffffff;
          padding: 10px 0;
          border-top: 1px solid #dddddd;
        }
        .input-box {
          max-width: 600px;
          margin: 0 auto;
        }
      </style>
    </head>
    <body>
      <div class="chat-container" id="chatContainer">
      </div>
      <div class="input-area">
        <div class="input-box container">
          <div class="input-group">
            <input type="text" class="form-control" id="questionInput" placeholder="请输入问题" autocomplete="off">
            <div class="input-group-append">
              <button class="btn btn-primary" id="sendBtn" type="button">发送</button>
            </div>
          </div>
        </div>
      </div>
      <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
      <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
      <script type="text/javascript">
        // 在页面中添加消息的函数，根据消息类型添加不同样式
        function appendMessage(message, type) {
          var bubbleClass = "";
          if (type === "user") {
            bubbleClass = "user-msg";
          } else {
            bubbleClass = "system-msg";
          }
          var bubble = $("<div></div>");
          bubble.addClass("chat-bubble");
          bubble.addClass(bubbleClass);
          var messageText = $("<span></span>").addClass("message-text").text(message);
          var timestamp = $("<span></span>").addClass("timestamp").text(getCurrentTime());
          bubble.append(messageText);
          bubble.append(timestamp);
          $("#chatContainer").append(bubble);
          $("#chatContainer").scrollTop($("#chatContainer")[0].scrollHeight);
        }
        // 获取当前时间，格式为 HH:MM
        function getCurrentTime() {
          var now = new Date();
          var hours = now.getHours();
          var minutes = now.getMinutes();
          if (minutes < 10) {
            minutes = '0' + minutes;
          }
          return hours + ':' + minutes;
        }
        // 文档加载完成后的事件绑定
        $(document).ready(function(){
          // 绑定发送按钮点击事件
          $("#sendBtn").click(function(){
            var question = $("#questionInput").val().trim();
            if (question === "") {
              return;
            }
            appendMessage(question, "user");  // 添加用户消息到聊天框
            $("#questionInput").val("");  // 清空输入框
            // 通过 Ajax 发送 POST 请求给服务端 /ask 接口
            $.ajax({
              url: "/ask",
              type: "POST",
              contentType: "application/json",
              data: JSON.stringify({question: question}),
              success: function(response) {
                appendMessage(response.answer, "system");  // 显示系统返回的答案
              },
              error: function(error) {
                appendMessage("请求出错，请稍后再试。", "system");  // 显示错误信息
                console.log("错误信息：", error);
              }
            });
          });
          // 绑定键盘回车事件，按回车发送消息
          $("#questionInput").keypress(function(event){
            if(event.which === 13) {
              $("#sendBtn").click();
              return false;
            }
          });
        });
      </script>
    </body>
    </html>
    """
    # 使用 Flask 的 render_template_string 渲染 HTML 页面
    return render_template_string(page_html)

# 主程序入口，启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

