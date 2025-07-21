# TF–IDF vs. BERT: A Detailed Comparison

This article provides a comprehensive comparison between the classic text vectorization method TF–IDF and the pretrained language model BERT. We cover mathematical definitions, model architectures, feature dimensions, computational efficiency, interpretability, and typical use cases. Complete example code based on Hugging Face models is provided to help readers understand their differences and suitable scenarios.

---

## 1. Mathematical Definitions

1. TF–IDF  
   - Suppose a corpus contains N documents. The number of documents in which term *t* appears is df(t).  
   - Inverse document frequency:  
     IDF(t) = log (N / (1 + df(t)))  
   - Term frequency TF(t, d) is the count of term *t* in document *d* (or a normalized variant).  
   - TF–IDF weight:  
     w(t, d) = TF(t, d) × IDF(t)  
   - Each document is represented as a sparse vector of dimension equal to the vocabulary size.

2. BERT  
   - Based on the Transformer encoder: input token embeddings + learnable positional encodings.  
   - Multi-Head Self-Attention captures global contextual dependencies.  
   - Pretraining tasks:  
     - Masked Language Modeling (MLM)  
     - Next Sentence Prediction (NSP)  
   - Outputs context-dependent dense vectors (default dimension 768 or 1024).

---

## 2. Model Architecture & Feature Dimensions

| Characteristic       | TF–IDF                                 | BERT                                       |
|----------------------|----------------------------------------|--------------------------------------------|
| Representation Type  | Sparse real-valued vector              | Dense real-valued vector                   |
| Dimension            | Vocabulary size (tens of thousands+)   | 768 / 1024                                 |
| Word Order/Context   | Discards order and context             | Bidirectional, context-sensitive           |
| Interpretability     | High                                   | Relatively low                             |

---

## 3. Computational Efficiency & Resource Usage

- TF–IDF  
  - Time complexity: O(#documents × average document length)  
  - Memory usage: scales linearly with vocabulary size and number of documents  
  - Can usually run in real time on CPU for vectorization and similarity search

- BERT  
  - Pretraining: requires massive corpora, multiple GPUs, and days of training  
  - Inference (encoding one sequence): O(L² × d), where L is sequence length and d is hidden size  
  - Typically needs GPU acceleration; on CPU, batch inference can be slow

---

## 4. Interpretability

- TF–IDF: transparent weight calculation, easy to identify “key” terms and their impacts  
- BERT: involves multiple layers of attention and nonlinear transformations, making it hard to interpret directly

---

## 5. Typical Use Cases

- TF–IDF  
  - Text retrieval and similarity  
  - Feature engineering for linear classifiers and clustering  
  - Resource-constrained scenarios

- BERT  
  - Complex text classification (sentiment analysis, intent detection)  
  - Sequence labeling (NER, tokenization)  
  - Question answering, text generation  
  - Tasks requiring deep semantic understanding and long-range dependencies

---

## 6. Code Examples

### 6.1 TF–IDF in Practice

Build a TF–IDF matrix with scikit-learn and compute cosine similarities between a query and documents.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load documents
docs = [
    "Machine learning can automatically extract models from data.",
    "Deep learning is a branch of machine learning that uses multi-layer neural networks.",
    "TF IDF is a text representation method based on term frequency and inverse document frequency."
]

# 2. Build the TF–IDF vectors
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    norm='l2'
)
tfidf_matrix = vectorizer.fit_transform(docs)

# 3. Vectorize the query & compute similarity
query = "deep neural network model"
query_vec = vectorizer.transform([query])
scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

# 4. Output results
rank = np.argsort(-scores)
print(f"Query: {query}\n")
for idx in rank:
    print(f"Document {idx} | Similarity: {scores[idx]:.4f} | Text: {docs[idx]}")
```

### 6.2 BERT Vectorization Example

Use the Hugging Face model `nlptown/bert-base-multilingual-uncased-sentiment` to extract sentence embeddings.

```python
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# 1. Prepare a QA knowledge base
qa_list = [
    {"question": "How to implement binary search in Python?",
     "answer": "Define left and right pointers on the sorted array and loop until you find the target or the interval is empty."},
    {"question": "What is TF–IDF?",
     "answer": "TF–IDF is a text representation method based on term frequency and inverse document frequency, used for information retrieval and feature extraction."},
    {"question": "How to load a Hugging Face pretrained model?",
     "answer": "Use `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained` to load the tokenizer and model."},
    {"question": "How to disable gradient calculation in PyTorch?",
     "answer": "Use the `with torch.no_grad():` context manager during inference to disable gradients."},
]

questions = [item["question"] for item in qa_list]

# 2. Load BERT tokenizer and model
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# 3. Precompute embeddings for the QA questions (using [CLS] pooling)
with torch.no_grad():
    encoded = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**encoded)
    qa_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

# 4. Define a retrieval function
def retrieve_answer(user_question, top_k=1):
    enc = tokenizer([user_question], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    q_vec = out.last_hidden_state[:, 0, :].cpu().numpy()
    sims = cosine_similarity(q_vec, qa_vecs).flatten()
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for idx in idxs:
        results.append({
            "matched_question": qa_list[idx]["question"],
            "answer": qa_list[idx]["answer"],
            "score": float(sims[idx])
        })
    return results

# 5. Example usage
test_questions = [
    "How to do binary search in a sorted array?",
    "How to load a model with transformers?",
    "How to turn off gradients in PyTorch?"
]

for uq in test_questions:
    hits = retrieve_answer(uq, top_k=1)
    hit = hits[0]
    print(f"User input : {uq}")
    print(f"Matched Q  : {hit['matched_question']}")
    print(f"Answer     : {hit['answer']}")
    print(f"Similarity : {hit['score']:.4f}")
    print("-" * 50)
```

---

## 7. Summary Comparison

- Efficiency & Memory: TF–IDF has a clear advantage  
- Representation & Downstream Performance: BERT excels  
- Suggested Use Cases:  
  - Lightweight retrieval, feature engineering → TF–IDF  
  - Deep understanding, complex tasks → BERT  

Choose the method that best fits your business needs and resource constraints when building your text processing pipeline.
