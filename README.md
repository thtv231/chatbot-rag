# 🤖 Chatbot RAG Đa Mô Hình (LangChain + Streamlit)

Ứng dụng chatbot thông minh cho phép **hỏi đáp trên tài liệu PDF** bằng cách kết hợp **RAG (Retrieval-Augmented Generation)** với nhiều **LLM khác nhau** như:

- 🧠 **Groq (Llama 3.1 8B)**
- ⚡ **Gemini (2.5 Flash)**
- 🧩 **Ollama (Gemma3:8B)**

---

## 🚀 Tính năng chính

✅ **Hỗ trợ đa mô hình LLM** — linh hoạt chọn model Groq, Gemini hoặc Ollama.  
✅ **Upload và hỏi PDF** — sử dụng **Ollama Embeddings (nomic-embed-text)** để lưu vector FAISS.  
✅ **RAG Pipeline tự động** — kết hợp truy vấn lịch sử hội thoại và sinh phản hồi theo ngữ cảnh.  
✅ **Giao diện Streamlit** — dễ sử dụng, có thể triển khai nhanh qua Streamlit Cloud hoặc local.

---

## 🧱 Cấu trúc thư mục

```bash
chatbot/
│
├── rag.py                # Mã nguồn chính (Streamlit app)
├── requirements.txt      # Thư viện cần thiết
├── .env                  # Chứa API keys (không commit)
├── .gitignore
└── README.md
