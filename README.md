# Production-Ready RAG AI Assistant

A full-stack, enterprise-grade AI Assistant designed for scalable local deployment. This system integrates a **Retrieval-Augmented Generation (RAG)** pipeline, **Function/Tool Calling**, and **Pydantic Structured Output Validation** using **Llama 3.1 (8B)** served via **Ollama**, backed by **ChromaDB** for vector retrieval, and presented through an interactive **Streamlit** web interface.

---

## 🏗️ Architecture & System Design

```
+-----------------------------------------------------------------------------------+
|                                 STREAMLIT WEB UI                                  |
|                 (Interactive Chat Interface & Session State Memory)               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        RELIABILITY & PERFORMANCE WRAPPER                          |
|             (Retry Mechanism, Latency Tracking, Error Degradation)                |
+-----------------------------------------------------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
+------------------------------------+        +------------------------------------+
|            RAG PIPELINE            |        |           TOOL CALLING             |
|  - Recursive Text Chunking         |        |  - System Metric Calculators       |
|  - ChromaDB Vector Store           |        |  - Dynamic Function Binding        |
|  - Ollama Dense Embeddings         |        |                                    |
+------------------------------------+        +------------------------------------+
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                LLM INFERENCE ENGINE                               |
|                         Llama 3.1:8b (Served via Ollama)                          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              STRUCTURED OUTPUT LAYER                              |
|                   Pydantic Schema Validation (Strict JSON Format)                 |
+-----------------------------------------------------------------------------------+
```

---

## ✨ Key Features

- **Local Inference Engine:** High-throughput, offline execution using **Llama 3.1 (8B)** via Ollama—ensuring zero data leakage and complete data privacy.
- **RAG Architecture:** High-density vector semantic search utilizing **ChromaDB** and `OllamaEmbeddings`.
- **Dynamic Tool Calling:** Native function binding using `@tool` decorators for external execution (e.g., system uptime calculations).
- **Structured JSON Validation:** Rigid response formatting using **Pydantic** models to return validated JSON objects with explicit metadata (summary, confidence score, source citations).
- **Interactive Web Interface:** Modern, responsive chat UI built with **Streamlit** featuring full conversation session memory.
- **Production Engineering & Fault Tolerance:**
  - Automated exponential retry policy for model inference calls.
  - Latency and compute execution metric reporting per response.
  - Resource caching (`@st.cache_resource`) for fast vector index loads.
  - Containerized deployment configuration via `Dockerfile` and `docker-compose.yml`.

---

## 🛠️ Tech Stack

- **Framework:** LangChain (`langchain-ollama`, `langchain-chroma`, `langchain-core`)
- **LLM & Embeddings:** Llama 3.1:8b (via Ollama)
- **Vector Database:** ChromaDB
- **User Interface:** Streamlit
- **Validation:** Pydantic v2
- **Containerization:** Docker & Docker Compose

---

## 📁 Repository Structure

```
.
├── app.py                  # Main Streamlit UI & Backend RAG/LLM Logic
├── knowledge_base.txt      # Source documentation ingested into ChromaDB
├── Dockerfile              # Docker image build configuration
├── docker-compose.yml      # Multi-container orchestration config
├── requirements.txt        # Python dependency specifications
└── README.md               # Project documentation
```

---

## 🚀 Getting Started (Local Setup)

### Prerequisites

1. **Python 3.10+** installed.
2. **Ollama** installed and running locally.
   - Install from [ollama.com](https://ollama.com)
   - Pull the model:
     ```bash
     ollama pull llama3.1:8b
     ```

### Installation

1. **Clone the repository / navigate to project folder:**
   ```bash
   cd W15-Productionizing-AI-Assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually install: `pip install -U streamlit langchain-ollama langchain-chroma chromadb pypdf pydantic`)*

4. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## 🐳 Containerized Deployment (Docker)

To run the application inside a containerized environment using Docker Compose:

1. Ensure Ollama is running on your host machine (`ollama serve`).
2. Build and launch the container stack:
   ```bash
   docker compose up --build
   ```
3. Access the application at `http://localhost:8501`.

---

## 🧪 Verification & Testing Scenarios

1. **RAG Vector Knowledge Retrieval:**
   - **Query:** `What database is used in this system?`
   - **Expected Result:** Extracts `ChromaDB` from `knowledge_base.txt`, attaches confidence ratings, and reports execution latency.

2. **Tool Execution:**
   - **Query:** `Calculate system uptime for 10 days active`
   - **Expected Result:** Binds and executes the internal `calculate_system_uptime` function to compute total operating hours (240 hours) with uptime metrics.

3. **Error Resilience:**
   - Stop the local Ollama server and submit a query.
   - **Expected Result:** Triggers the retry handler and presents a clean error notice in the UI without crashing the application.
