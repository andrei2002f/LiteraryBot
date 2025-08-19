# 📚 Literary RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot specialized in **Romanian literature**.  
It combines **semantic search** in Elasticsearch with **large language models (LLMs)** via Ollama, and includes a **Flask web frontend** for interactive use.

---

## ✨ Features

- 🔍 **Hybrid Search**: dense vector similarity (SentenceTransformers) + keyword scoring (BM25).  
- 📝 **Context Filtering**: Romanian abbreviation-aware sentence tokenizer with highlight-based context compression.  
- 🤖 **LLM Integration**: Supports multiple Ollama models:
  - `llama3.1:70b`
  - `llama3.3:latest`
  - `deepseek-r1:70b`
  - `qwen2.5:72b`
  - `llama4:16x17b`
- 📊 **Evaluation Framework**: Accuracy, recall@k, latency, and semantic similarity scoring on curated Q&A sets.  
- 🗂 **Unified Index**: Authors and publications indexed together, enriched with dynamic keyword extraction.  
- 🌐 **Flask UI**: Minimal chat interface with conversation history and optional model selector.

---

## ⚡ Quickstart

```bash
# 1) Create virtual environment & install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# 2) Start Elasticsearch (default http://localhost:9200) and index the data
python src/indexer.py

# 3) Launch the web app (Flask on http://localhost:5000)
python app.py
