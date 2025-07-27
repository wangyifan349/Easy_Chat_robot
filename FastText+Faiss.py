import fasttext
import faiss
import numpy as np

# 1. 加载 FastText 预训练模型
# 请确保你已经下载了 FastText 预训练的模型文件 (例如：cc.en.300.bin)
ft_model = fasttext.load_model('cc.en.300.bin')  # 你需要指定正确的模型文件路径

# 2. 定义一些预先设定的问答对
qa_data = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who is the president of the United States?", "The president of the United States is Joe Biden."),
    ("What is the largest ocean?", "The largest ocean is the Pacific Ocean."),
    ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second."),
    ("Where is the Great Wall of China?", "The Great Wall of China is located in China."),
]

# 3. 将答案转化为向量并添加到 FAISS 索引中
qa_vectors = []
for question, answer in qa_data:
    # 将每个答案转化为 FastText 词向量
    answer_vec = ft_model.get_sentence_vector(answer)
    qa_vectors.append(answer_vec)

qa_vectors = np.array(qa_vectors).astype('float32')

# 4. 创建 FAISS 索引
index = faiss.IndexFlatL2(300)  # 300是 FastText 模型生成的词向量的维度
index.add(qa_vectors)  # 将所有答案的词向量添加到索引中

# 5. 定义一个查询函数来处理用户输入的查询
def answer_question(query):
    # 将用户输入的问题转化为向量
    query_vec = ft_model.get_sentence_vector(query).reshape(1, -1).astype('float32')

    # 使用 FAISS 查找与问题最相似的答案（返回一个最相似的答案）
    D, I = index.search(query_vec, k=1)  # k=1，表示返回最相似的一个答案

    # 输出查询结果
    print(f"Question: {query}")
    print(f"Answer: {qa_data[I[0][0]][1]}")  # 根据索引获取对应的答案

# 6. 测试查询
if __name__ == "__main__":
    while True:
        # 获取用户输入
        query = input("Ask a question (or type 'exit' to quit): ").strip()
        # 输入 "exit" 时退出程序
        if query.lower() == "exit":
            print("Goodbye!")
            break
        # 获取并输出答案
        answer_question(query)
