# Product Requirements Document (PRD): Local AI Advisor (v3)

## 1. Product Overview
**Name:** Local AI Advisor  
**Concept:** A privacy-first, locally hosted AI chat interface acting as a technical mentor, document analyzer, and career coach. It leverages local LLMs, a Retrieval-Augmented Generation (RAG) pipeline, and real-time web scraping.  
**Value Proposition:** Intelligence of premium cloud tools with 100% data privacy, optimized for Apple Silicon (M4, 16GB RAM) to process proprietary code, coursework, and job postings on-device.

## 2. Target Audience
**Keith Speirs** Data Intern and someone who needs advise from personas he made

## 3. Core Features (MVP)
| Feature | Description | Priority |
| :--- | :--- | :--- |
| **Local LLM Integration** | Connection to Ollama/LM Studio via OpenAI-compatible API. | High |
| **Persona Selection** | Sidebar to switch system prompts (Data Engineer, Architect, Career Coach). | High |
| **Document RAG** | Upload PDFs/Text; system chunks and embeds for semantic search. | High |
| **Real-Time Scraper** | URL input field to strip HTML and feed raw text into context. | High |
| **Local Vector DB** | ChromaDB for storing document and web embeddings locally. | High |
| **Session Memory** | Continuous chat history retention within the thread. | High |
| **Streaming** | Token-by-token response for a zero-lag feel. | Medium |

## 4. Technical Architecture
### 4.1 Hardware & Infrastructure
*   **OS:** macOS (Apple Silicon M4, 16GB RAM).
*   **Deployment:** Localhost (Streamlit: 8501, Ollama: 11434).

### 4.2 Software Stack
*   **UI:** Streamlit (Python).
*   **Inference:** Ollama or LM Studio.
*   **Orchestration:** LangChain / LangGraph.
*   **Vector DB:** ChromaDB.
*   **Parsing:** BeautifulSoup4 / PyPDF2.

### 4.3 Project Structure
```text
AI Advisory/
├── src/
│   └── app.py          # Main Streamlit application
├── data/
│   ├── db/             # ChromaDB vector store
│   └── uploads/        # Uploaded PDF/Text files
├── assets/             # Images and static assets
├── venv/               # Python virtual environment
├── requirements.txt    # Project dependencies
└── PRD.md              # Product Requirements Document
```

## 5. Primary Use Cases
1.  **Job Matcher:** Scrape job URL + Upload Resume -> AI suggests rewrites and identifies missing keywords.
2.  **Coursework Prep:** Upload syllabi/slides -> AI generates quizzes and summaries offline.
3.  **Arch Review:** Upload DB schema -> "Data Engineering" persona provides optimized SQL/Node.js logic.

## 6. Future Enhancements
*   Persistent SQLite history for cross-session continuity.
*   Automated Mock Interviews (Persona-driven).
*   Portfolio analysis pipeline.
