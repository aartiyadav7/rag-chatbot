# 🤖 RAG Chatbot

A production-ready Retrieval-Augmented Generation chatbot that answers
questions from any PDF or website — with voice input and text-to-speech.

## Features
- 📄 PDF and URL ingestion
- 🔍 Semantic search with ChromaDB + HuggingFace embeddings
- 🤖 LLaMA 3.1 answers via Groq (free and fast)
- 🔊 Text-to-Speech using Google gTTS
- 🎙️ Voice input using browser Speech-to-Text
- 💬 Multi-turn chat with cited sources

## Tech Stack
| Layer | Technology |
|---|---|
| Embeddings | all-MiniLM-L6-v2 (local, no GPU needed) |
| Vector Store | ChromaDB |
| LLM | LLaMA 3.1 8B via Groq API |
| Framework | LangChain + Streamlit |
| TTS | Google gTTS |
| STT | Google SpeechRecognition |

## Run Locally
```bash
git clone https://github.com/aartiyadav7/rag-chatbot
cd rag-chatbot
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
python -m streamlit run app/streamlit_app.py
```

## Project Structure
```
src/
├── ingestion/     # PDF + URL loaders, chunker
├── embeddings/    # HuggingFace local embedder
├── retrieval/     # ChromaDB vector store
├── generation/    # Groq LLM + prompt templates
└── pipeline/      # End-to-end ingest + query
app/
└── streamlit_app.py  # Streamlit UI with TTS + STT
```