from  flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

app = Flask(__name__)

def load_qa_pairs():
    # 也可以从 qa_pairs.json 文件加载多行字符串问答
    if os.path.exists("qa_pairs.json"):
        with open("qa_pairs.json", encoding="utf-8") as f:
            return json.load(f)
    else:
        return [
            {
                "question": "如何用Python打印一段代码？",
                "answer": """你可以这样写：

```python
print("Hello, World!")
```

这段代码会输出一行文本。
"""
            },
            {
                "question": "今天天气怎么样？",
                "answer": "今天阳光明媚，适合出去散步。"
            }
        ]

qa_pairs = load_qa_pairs()
questions = [q['question'] for q in qa_pairs]

model = SentenceTransformer('uer/sbert-base-chinese-nli')
question_embeddings = model.encode(questions, convert_to_numpy=True)

def search_by_embedding(user_question):
    user_emb = model.encode([user_question], convert_to_numpy=True)[0]
    user_norm = np.linalg.norm(user_emb)
    norms = np.linalg.norm(question_embeddings, axis=1)
    similarities = np.dot(question_embeddings, user_emb) / (norms * user_norm + 1e-10)
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    return best_idx, best_score

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "请输入问题"}), 400
    best_idx, best_score = search_by_embedding(user_question)
    threshold = 0.5
    if best_score < threshold:
        answer = "抱歉，我无法理解您的问题，请尝试换个说法。"
    else:
        answer = qa_pairs[best_idx]["answer"]
    return jsonify({"answer": answer})

@app.route("/")
def index():
    # 用 marked.js 前端渲染 markdown，配合 highlight.js 做代码高亮
    page_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8" />
    <title>在线问答机器人</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/css/bootstrap.min.css" />
    <!-- highlight.js 样式 -->
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <style>
        body {
            background: #f6f8fa;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            padding-bottom: 80px; /* 底部输入框留空 */
        }
        .chat-container {
            max-width: 700px;
            margin: 30px auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            height: 500px;
            overflow-y: auto;
            box-shadow: 0 2px 9px rgba(0,0,0,0.1);
        }
        .chat-bubble {
            padding: 12px 18px;
            border-radius: 20px;
            margin-bottom: 12px;
            max-width: 80%;
            position: relative;
            font-size: 15px;
            line-height: 1.5;
            white-space: pre-wrap; /* 保留换行和空格 */
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            word-wrap: break-word;
        }
        .user-msg {
            background: #dcf8c6;
            float: right;
            text-align: right;
            border-bottom-right-radius: 4px;
            animation: fadeInRight 0.3s ease;
        }
        .system-msg {
            background: #f1f0f0;
            float: left;
            border-bottom-left-radius: 4px;
            animation: fadeInLeft 0.3s ease;
        }
        .timestamp {
            font-size: 10px;
            color: #999;
            position: absolute;
            bottom: -16px;
            right: 15px;
            user-select: none;
        }
        .input-area {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: #fff;
            border-top: 1px solid #ddd;
            padding: 12px 0;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
            z-index: 100;
        }
        .input-box {
            max-width: 700px;
            margin: 0 auto;
            padding: 0 15px;
        }
        #questionInput {
            font-size: 16px;
            height: 40px;
            border-radius: 30px !important;
            padding-left: 20px;
            box-shadow: none !important;
        }
        #sendBtn {
            border-radius: 30px !important;
            min-width: 80px;
            font-weight: 600;
        }
        @keyframes fadeInRight {
            from {opacity: 0; transform: translateX(40px);}
            to {opacity: 1; transform: translateX(0);}
        }
        @keyframes fadeInLeft {
            from {opacity: 0; transform: translateX(-40px);}
            to {opacity: 1; transform: translateX(0);}
        }
        /* 清除浮动 */
        .clearfix::after {
            content: "";
            clear: both;
            display: table;
        }
        /* highlight.js 代码块美化 */
        pre {
            background: #282c34;
            color: #abb2bf;
            padding: 12px 15px;
            border-radius: 8px;
            font-size: 13px;
            overflow-x: auto;
        }
        code {
            font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
        }
    </style>
</head>
<body>
    <div class="chat-container" id="chatContainer" aria-live="polite" aria-atomic="false"></div>
    
    <div class="input-area">
        <div class="input-box container">
            <div class="input-group">
                <input type="text" class="form-control" placeholder="请输入问题" id="questionInput" aria-label="请输入您的问题" autocomplete="off" />
                <div class="input-group-append">
                    <button class="btn btn-primary" id="sendBtn" aria-label="发送消息">发送</button>
                </div>
            </div>
        </div>
    </div>

    <!-- 引入 marked.js 用于 markdown 转 HTML -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- 引入 highlight.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>
        // 高亮 markdown 中代码块
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                  return hljs.highlight(code, {language: lang}).value;
                }
                return hljs.highlightAuto(code).value;
            }
        });

        function appendMessage(text, role) {
            var container = document.getElementById("chatContainer");
            // 消息外层div
            var bubble = document.createElement("div");
            bubble.classList.add("chat-bubble", role === "user" ? "user-msg" : "system-msg", "clearfix");

            if(role === "user") {
                // 用户消息直接转义避免 XSS
                bubble.textContent = text;
            } else {
                // 系统消息以 markdown 渲染
                bubble.innerHTML = marked.parse(text);
            }

            // 时间戳
            var timestamp = document.createElement("span");
            timestamp.className = "timestamp";
            timestamp.textContent = getCurrentTime();
            bubble.appendChild(timestamp);

            container.appendChild(bubble);
            container.scrollTop = container.scrollHeight;
        }

        function getCurrentTime() {
            var d = new Date();
            var h = d.getHours();
            var m = d.getMinutes();
            return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m;
        }

        function sendMessage() {
            var input = document.getElementById("questionInput");
            var question = input.value.trim();
            if(question === "") return;
            appendMessage(question, "user");
            input.value = "";
            input.focus();

            fetch("/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question: question})
            }).then(res => res.json())
            .then(data => {
                if(data.error){
                    appendMessage("发生错误：" + data.error, "system");
                } else {
                    appendMessage(data.answer, "system");
                }
            })
            .catch(err => {
                appendMessage("请求失败，请稍后再试。", "system");
                console.error(err);
            });
        }

        document.getElementById("sendBtn").addEventListener("click", sendMessage);
        document.getElementById("questionInput").addEventListener("keypress", function(e){
            if(e.key === "Enter") {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """
    return render_template_string(page_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
