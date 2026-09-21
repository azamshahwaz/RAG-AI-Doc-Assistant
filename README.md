# RAG Assistant (AI Document Assistant)

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that lets users chat with their documents via text or voice, with web search fallback and per-user private chat history.

## Features

- **Multi-format document upload** (up to 10 files): PDF, DOCX/DOC, PPTX/PPT, XLSX/XLS/CSV, TXT, RTF, ODT, Markdown, JSON, HTML
- **Document Q&A** using LangChain + FAISS vector store with HuggingFace sentence-transformer embeddings
- **Web search fallback** (DuckDuckGo) when the answer isn't found in uploaded documents
- **Per-document-type summary/analysis report** generation
- **Voice assistant** — Whisper (faster-whisper) based capture with noise calibration, VAD, and hallucination filtering; answers routed through the same document → web → both pipeline with sources
- **Authentication** — SQLite-based, PBKDF2 password hashing, persistent cookie sessions with a 2-hour sliding inactivity timeout
- **Per-user private chat history**, with PDF/TXT conversation export
- **LLM**: Groq API (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)

## Tech Stack

Python · Streamlit · LangChain · FAISS · HuggingFace Embeddings · Groq API · faster-whisper · SQLite

## Project Structure

```
├── app.py                # Main Streamlit app / entry point
├── auth.py                # Authentication logic (PBKDF2, sessions)
├── auth_ui.py              # Login/signup UI components
├── chat_manager.py           # Chat history management
├── ingest.py               # Document ingestion & preprocessing
├── rag.py / rag_engine.py       # Core RAG pipeline (retrieval + generation)
├── qa_handler.py             # Q&A orchestration (LLM calls via Groq)
├── analysis_report.py          # Per-document summary/report generation
├── voice_assistant.py          # Voice input/output (Whisper + TTS)
├── tts.py                 # Text-to-speech utilities
├── image_generator.py          # Image generation utilities
├── share_utils.py             # Chat/report sharing utilities
├── session_state.py           # Streamlit session state helpers
├── conversations/             # Per-user saved chat history (JSON)
└── requirements.txt
```

## Setup

1. **Clone and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables** — create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key   # only needed for image_generator.py
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Notes

- `users.db` stores user accounts and hashed passwords.
- `conversations/<username>/*.json` stores each user's chat history for the sidebar and export features.
- A rebuild with a separate FastAPI backend (PostgreSQL + SQLAlchemy + Alembic) and dedicated frontend is in progress to replace this Streamlit version.

## License

Personal/academic project — no license specified.
