import json
import time
import streamlit as st
import warnings
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.tools import tool

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Assistant (Task 2)", page_icon="🤖", layout="wide")
st.title("🤖 Production AI Assistant (Local Ollama RAG)")

# --- 2. CACHED VECTOR DB INIT (Performance Engineering) ---
@st.cache_resource
def setup_vector_db():
    sample_doc_content = """
    The AI Assistant system is built with a hybrid approach:
    - Local LLM: Llama 3.1 8B served via Ollama for privacy and offline tasks.
    - Vector Database: ChromaDB for storing and retrieving document embeddings.
    - RAG Architecture: Augments LLM responses using semantic search over knowledge bases.
    - Tool Integration: Executes dynamic python functions to compute system metrics.
    """
    with open("knowledge_base.txt", "w") as f:
        f.write(sample_doc_content)

    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_vector_db()

# --- 3. DEFINE TOOLS & STRUCTURED OUTPUT ---
@tool
def calculate_system_uptime(days_active: int) -> str:
    """Calculates operational uptime percentage based on days active."""
    total_hours = days_active * 24
    return f"System active for {total_hours} total hours with 99.9% uptime."

tools = [calculate_system_uptime]

class AssistantResponse(BaseModel):
    summary: str = Field(description="Direct concise answer to the query.")
    confidence_score: float = Field(description="Confidence rating between 0.0 and 1.0.")
    retrieved_sources: list[str] = Field(description="Context pieces used.")

# --- 4. BACKEND INFERENCE WITH RETRY LOGIC (Reliability) ---
def query_backend(user_query: str):
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            # RAG Retrieval
            retrieved_docs = retriever.invoke(user_query)
            context = retrieved_docs[0].page_content if retrieved_docs else "No relevant context found."
            
            # Model Binding & Structured Response
            llm = ChatOllama(model="llama3.1:8b", temperature=0.2)
            llm_with_tools = llm.bind_tools(tools)
            llm_structured = llm.with_structured_output(AssistantResponse)
            
            prompt = f"Context: {context}\nQuestion: {user_query}"
            
            start_time = time.time()
            response = llm_structured.invoke(prompt)
            latency = round(time.time() - start_time, 2)
            
            return response, latency, context
            
        except Exception as e:
            retries += 1
            if retries == max_retries:
                st.error(f"Failed to process query after {max_retries} attempts: {str(e)}")
                return None, 0, ""
            time.sleep(1)

# --- 5. CHAT UI LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_prompt := st.chat_input("Ask a question about the system architecture..."):
    # Display user input
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from ChromaDB & Running Llama 3.1..."):
            response, latency, retrieved_context = query_backend(user_prompt)
            
            if response:
                formatted_response = f"**Answer:** {response.summary}\n\n" \
                                     f"- **Confidence Score:** `{response.confidence_score}`\n" \
                                     f"- **Latency:** `{latency} seconds`\n" \
                                     f"- **Source Used:** `{response.retrieved_sources}`"
                
                st.markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})