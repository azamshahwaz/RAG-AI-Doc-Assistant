# 🤖 RAG Assistant — AI Document Assistant

An AI-powered **Research & Document Assistant** built with **Streamlit and Retrieval-Augmented Generation (RAG)**.

RAG Assistant allows users to upload and chat with multiple documents, perform web research, generate document analysis reports, interact using voice, and maintain **private user-specific conversations**.

---

## ✨ Features

* 📄 **Multi-Format Document Upload** — Upload up to 10 documents including PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, CSV, TXT, RTF, ODT, Markdown, JSON and HTML.
* 🧠 **RAG-Based Document Q&A** — Ask questions about uploaded documents using LangChain, FAISS and Hugging Face embeddings.
* 🔎 **Intelligent Query Routing** — Automatically routes queries to document retrieval, web search, or both.
* 🌐 **Web Search Fallback** — Uses DuckDuckGo search when relevant information is not available in uploaded documents.
* 📚 **Source-Aware Answers** — Provides relevant document sources such as file, page, slide or sheet references.
* 📊 **AI Document Analysis** — Generates document-specific summaries and structured analysis reports.
* 💬 **Private Chat History** — Maintains user-specific conversations with persistent chat history.
* 📤 **Conversation Export** — Export conversations in PDF or TXT format.
* 🔐 **Authentication & Sessions** — SQLite-based authentication with PBKDF2 password hashing and persistent sessions.
* 🎙️ **Voice Assistant** — Voice-based questions and commands using Faster-Whisper with noise calibration, VAD and hallucination filtering.
* 🌐 **Website Launcher** — Open websites such as Google, YouTube, Gmail, GitHub, LinkedIn, Naukri and other supported websites using **voice or chat commands**.
* 🔊 **Text-to-Speech** — Converts AI responses into speech using Edge TTS and pyttsx3.
* 🖼️ **AI Image Generation** — Generate images from text prompts using the OpenAI Image API.
* ⚡ **Performance & Reliability** — Resource caching, response-time tracking, API rate-limit handling and error handling.

---

## 🏗️ How It Works

```text
                    User
                     │
            ┌────────┴────────┐
            │                 │
          Text              Voice
            │                 │
            └────────┬────────┘
                     ▼
              Query Processing
                     │
                     ▼
              Intent / Routing
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Documents     Web        Both
          │          │          │
          └──────────┼──────────┘
                     ▼
                Groq LLM
                     │
                     ▼
            Answer + Sources
```

### Document RAG Pipeline

```text
Upload Documents
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Hugging Face Embeddings
       ↓
FAISS Vector Store
       ↓
Semantic Retrieval
       ↓
Relevant Context
       ↓
Groq LLM
       ↓
Grounded Answer
```

---

## 🧰 Tech Stack

| Category            | Technologies                                      |
| ------------------- | ------------------------------------------------- |
| Language            | Python                                            |
| UI                  | Streamlit                                         |
| RAG                 | LangChain, FAISS                                  |
| Embeddings          | Hugging Face Sentence Transformers                |
| LLM                 | Groq API                                          |
| Web Search          | DuckDuckGo / DDGS                                 |
| Voice               | Faster-Whisper                                    |
| Voice Matching      | RapidFuzz                                         |
| Text-to-Speech      | Edge TTS, pyttsx3                                 |
| Authentication      | SQLite, PBKDF2                                    |
| Image Generation    | OpenAI Image API                                  |
| Document Processing | PyPDF, python-docx, Pandas, OpenPyXL, python-pptx |
| Configuration       | python-dotenv                                     |

---

## 📁 Project Structure

```text
RAG Assistant/
│
├── app.py                    # Main Streamlit application
│
├── auth.py                   # Authentication and session logic
├── auth_ui.py                # Login and signup UI
├── session_state.py          # Streamlit session-state helpers
│
├── chat_manager.py           # Chat history management
├── share_utils.py            # Chat/report sharing and export utilities
│
├── ingest.py                 # Document loading and preprocessing
├── rag.py                    # RAG functionality
├── rag_engine.py             # Retrieval and generation pipeline
├── qa_handler.py             # Question-answering orchestration
│
├── analysis_report.py        # Document analysis/report generation
│
├── voice_assistant.py        # Voice assistant functionality
├── tts.py                    # Text-to-speech utilities
│
├── image_generator.py        # AI image generation
│
├── conversations/            # Per-user conversation data
├── users.db                  # SQLite authentication database
│
├── requirements.txt          # Python dependencies
├── .env                      # API credentials (not committed)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

`OPENAI_API_KEY` is required only for the image-generation functionality.

> **Never commit your `.env` file or API keys to GitHub.**

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 💡 Example Usage

### 📄 Document Q&A

Upload a research paper and ask:

```text
Summarize this document.
```

```text
What methodology was used?
```

```text
What are the key findings?
```

### 🌐 Web Research

Ask:

```text
What are the latest developments in this technology?
```

If the required information is not available in the uploaded documents, the assistant can use web search.

### 🎙️ Voice Commands

You can interact with the assistant using voice commands such as:

```text
Open YouTube
Open Google
Open Gmail
Open GitHub
Open LinkedIn
Open Naukri
```

The assistant identifies the requested website and opens it in the browser.

### 💬 Chat Commands

The same type of supported website commands can also be given through the text chat interface.

---

## 🔐 Authentication & Privacy

The application provides user-specific authentication and data isolation.

### Authentication

* SQLite-based user database
* PBKDF2 password hashing
* Secure password storage
* Persistent sessions
* Session expiration
* User logout
* User-specific conversations

### User Data

```text
users.db
    │
    └── User accounts & password hashes

conversations/
    │
    ├── User 1
    │    └── chat history
    │
    └── User 2
         └── chat history
```

Each user's conversation history is stored separately.

---

## 📊 Document Analysis

RAG Assistant can generate AI-powered analysis for different document types, including:

* Research Papers
* Resumes / CVs
* Project Reports
* Technical Documentation
* Academic Documents
* Datasets
* Business Documents
* Presentations
* General Documents

The analysis system generates structured summaries and insights based on the uploaded content.

---

## 🎙️ Voice Assistant Pipeline

```text
Microphone
    ↓
Noise Calibration
    ↓
Voice Activity Detection
    ↓
Faster-Whisper
    ↓
Speech → Text
    ↓
Query Processing
    ↓
Document / Web / Both
    ↓
Groq LLM
    ↓
AI Response
    ↓
Text-to-Speech
```

The voice assistant also includes filtering to reduce unwanted speech-recognition hallucinations.

---

## ⚙️ Configuration

### Groq Models

The application uses Groq-hosted LLMs for AI responses.

Configured models include:

```text
llama-3.3-70b-versatile
llama-3.1-8b-instant
```

Model selection can be configured according to the application's requirements and available API limits.

---

## 🗃️ Data Storage

The current Streamlit version uses local storage:

```text
users.db
```

for authentication data and:

```text
conversations/
```

for user-specific chat history.

Vector data is maintained through the FAISS-based RAG pipeline.

---

## 🛡️ Security Notes

The application includes:

* PBKDF2 password hashing
* User-specific sessions
* Password validation
* Session expiration
* Environment-based API credentials
* Separation of user conversation data

For production deployment, additional security measures such as HTTPS, secure cookies, CSRF protection, centralized secret management and production-grade authentication should be considered.

---

## 🔮 Future Development

A separate backend architecture is currently being developed to move beyond the Streamlit-only implementation.

Planned architecture:

```text
Frontend
   ↓
FastAPI Backend
   ↓
PostgreSQL
   ↓
pgvector
   ↓
RAG / AI Services
```

Planned improvements include:

* FastAPI backend
* PostgreSQL database
* SQLAlchemy
* Alembic migrations
* pgvector-based retrieval
* Dedicated frontend
* Production-ready authentication
* Scalable deployment architecture

---

## 📌 Project Highlights

```text
✓ Multi-document RAG
✓ Semantic document search
✓ Web research
✓ Intelligent query routing
✓ Source-aware answers
✓ AI document analysis
✓ Persistent private chat history
✓ Voice-based interaction
✓ Website launcher through voice & chat
✓ Text-to-speech
✓ AI image generation
✓ Secure authentication
✓ Conversation export
✓ API rate-limit handling
```

---

## 📜 License

This is a **personal/academic project**. No open-source license has been specified.

---

## 👨‍💻 Author

**Shahwaz Azam**

Computer Science & Engineering — Data Science

---

⭐ If you find this project interesting, consider starring the repository.
