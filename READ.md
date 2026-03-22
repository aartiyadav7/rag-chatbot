# 🤖 RAG Chatbot — Ask Anything from PDFs & Websites

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=for-the-badge&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-orange?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A production-ready RAG chatbot with voice input, text-to-speech, and semantic search**

[🚀 Live Demo](#) • [📖 How It Works](#how-it-works) • [⚙️ Setup](#setup) • [🏗️ Architecture](#architecture)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Ingestion** | Upload any PDF and ask questions from it |
| 🌐 **URL Ingestion** | Paste any website URL and chat with its content |
| 🔍 **Semantic Search** | Finds the most relevant chunks using vector similarity |
| 🤖 **LLaMA 3.1** | Fast, free LLM answers via Groq API |
| 🔊 **Text-to-Speech** | Answers read aloud using Google gTTS |
| 🎙️ **Voice Input** | Ask questions by speaking into your browser mic |
| 💬 **Multi-turn Chat** | Full conversation history with cited sources |
| 🧠 **Local Embeddings** | No GPU needed — runs on CPU with HuggingFace |

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│              Streamlit App (TTS + STT + Chat)               │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────┐           ┌─────────────────┐
│ INGESTION       │           │ QUERY           │
│ PIPELINE        │           │ PIPELINE        │
│                 │           │                 │
│ PDF / URL       │           │ User Question   │
│     ↓           │           │     ↓           │
│ Chunker         │           │ Embed Query     │
│     ↓           │           │     ↓           │
│ Embedder        │           │ Retrieve Top-K  │
│     ↓           │           │     ↓           │
│ ChromaDB        │           │ Groq LLaMA 3.1  │
│ Vector Store    │           │     ↓           │
└─────────────────┘           │ Answer + Source │
                              └─────────────────┘
```

---

## 🔄 How It Works
```
1. INGEST                2. EMBED                3. RETRIEVE
─────────────           ─────────────           ─────────────
PDF or URL    →    Split into chunks  →   Store in ChromaDB
                   Embed with           Vector similarity
                   MiniLM-L6-v2         search (top-5)

4. GENERATE              5. RESPOND
─────────────           ─────────────
Build prompt   →    LLaMA 3.1 via     →   Answer + Sources
with context        Groq API               + TTS audio
```

---

## 🛠️ Tech Stack
```
┌──────────────┬─────────────────────────────────────────┐
│ Layer        │ Technology                               │
├──────────────┼─────────────────────────────────────────┤
│ UI           │ Streamlit 1.55                           │
│ LLM          │ LLaMA 3.1 8B via Groq API (free)        │
│ Embeddings   │ all-MiniLM-L6-v2 (HuggingFace, local)   │
│ Vector Store │ ChromaDB (persistent)                    │
│ Framework    │ LangChain + LangChain-Community          │
│ PDF Parser   │ PyMuPDF (fitz)                           │
│ Web Scraper  │ BeautifulSoup4                           │
│ TTS          │ Google gTTS                              │
│ STT          │ Google SpeechRecognition                 │
│ Language     │ Python 3.14                              │
└──────────────┴─────────────────────────────────────────┘
```

---

## 📁 Project Structure
```
rag-chatbot/
│
├── 📁 app/
│   └── streamlit_app.py        ← Main UI (TTS + STT + Chat)
│
├── 📁 src/
│   ├── 📁 ingestion/
│   │   ├── pdf_loader.py       ← PyMuPDF PDF parser
│   │   ├── web_loader.py       ← BeautifulSoup URL scraper
│   │   └── chunker.py          ← Recursive text splitter
│   │
│   ├── 📁 embeddings/
│   │   └── embedder.py         ← HuggingFace local embeddings
│   │
│   ├── 📁 retrieval/
│   │   └── vector_store.py     ← ChromaDB vector store
│   │
│   ├── 📁 generation/
│   │   ├── llm.py              ← Groq LLaMA 3.1 client
│   │   └── prompt_templates.py ← RAG system + user prompts
│   │
│   ├── 📁 pipeline/
│   │   ├── ingest_pipeline.py  ← End-to-end ingestion
│   │   └── query_pipeline.py   ← End-to-end query
│   │
│   └── 📁 utils/
│       ├── config.py           ← Pydantic settings
│       └── logger.py           ← Structured logging
│
├── 📁 tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
│
├── 📁 data/
│   ├── raw/                    ← Uploaded PDFs
│   └── processed/              ← ChromaDB vector store
│
├── .env.example                ← Environment template
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation
```bash
# 1. Clone the repo
git clone https://github.com/aartiyadav7/rag-chatbot
cd rag-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Environment Variables
```env
GROQ_API_KEY=gsk_your_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
CHROMA_DB_PATH=./data/processed/chroma_db
```

### Run
```bash
python -m streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🎯 Usage
```
1. Open the app at localhost:8501

2. In the sidebar:
   ├── Paste a URL  →  Click "Ingest URL"
   └── Upload PDF   →  Click "Ingest PDF"

3. Wait ~30 seconds for indexing

4. Ask questions in the chat box
   └── Or toggle 🎙️ Voice Input to speak

5. Get answers with cited sources
   └── Toggle 🔊 TTS to hear the answer
```

### Example URLs to try
```
https://en.wikipedia.org/wiki/Retrieval-augmented_generation
https://en.wikipedia.org/wiki/Large_language_model
https://huggingface.co/blog/rag
https://docs.python.org/3/tutorial/
```

---

## 🧪 Run Tests
```bash
# Test ingestion pipeline
python -m pytest tests/test_ingestion.py -v -s

# Test vector store + retrieval
python -m pytest tests/test_retrieval.py -v -s

# Test full end-to-end pipeline
python -m pytest tests/test_pipeline.py -v -s
```

---

## 🚀 Enhancements Roadmap

- [x] PDF ingestion
- [x] URL ingestion
- [x] ChromaDB vector store
- [x] Groq LLaMA 3.1 generation
- [x] Text-to-Speech (TTS)
- [x] Voice input (STT)
- [x] Multi-turn chat history
- [ ] Re-ranking with Cohere
- [ ] RAGAS evaluation suite
- [ ] Multi-document support
- [ ] Docker deployment
- [ ] FastAPI backend

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use this for your own projects!

---

<div align="center">

Built with ❤️ by [Aarti Yadav](https://github.com/aartiyadav7)

⭐ Star this repo if you found it helpful!

</div>