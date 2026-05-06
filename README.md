# 🧠 Custom RAG System with Custom GPT + Mistral

A fully custom Retrieval-Augmented Generation (RAG) system built from scratch using:

- Custom Transformer/GPT implementation
- FAISS vector search
- Hybrid retrieval
- Semantic embeddings
- Query expansion
- Local Mistral inference using Ollama
- Custom autoregressive text generation

This project demonstrates the internal mechanics behind modern LLM systems and production-style RAG pipelines.

---

# 🚀 Features

## 🔍 Retrieval-Augmented Generation (RAG)

- Document chunking with overlap
- Semantic embedding generation
- FAISS vector database
- Hybrid retrieval (semantic + keyword reranking)
- Query expansion
- Context-aware answer generation

---

## 🤖 Dual Generation Modes

### 1. Mistral Mode (Production-style)

Uses:
- Ollama
- Local Mistral model
- REST API inference

### 2. Custom GPT Mode

Uses:
- Fully custom Transformer implementation
- Multi-head self-attention
- Positional embeddings
- Token sampling
- GPT-2 weight loading

---

# 🧠 What This Project Demonstrates

This project was built to deeply understand how modern LLM systems work internally.

Instead of only calling APIs, this system implements:

- Transformer architecture
- Attention mechanism
- Autoregressive generation
- Vector retrieval systems
- Embedding pipelines
- Hybrid search
- Prompt engineering
- Local LLM orchestration

---

# 🏗️ System Architecture

```text
Raw Documents
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Semantic Retrieval
      ↓
Hybrid Re-ranking
      ↓
Prompt Construction
      ↓
LLM Generation
      ↓
Final Answer
```

---

# 📂 Project Structure

```text
custom-rag-system/
│
├── data/
│   ├── big.txt
│   └── The_Verdict.txt
│
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   ├── generator.py
│   ├── generator_mistral.py
│
├── model/
│   └── custom_gpt.py
│
├── gpt_download3.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Technologies Used

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Embeddings | Sentence Transformers |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | FAISS |
| Local LLM Runtime | Ollama |
| External Model | Mistral |
| Tokenization | tiktoken |
| GPT Weights | GPT-2 124M |
| Backend Logic | Python |

---

# 🧩 Core Components

## 1. Chunking

Splits large documents into overlapping chunks for retrieval.

```python
chunk_size=900
overlap=250
```

Purpose:
- preserve context continuity
- improve retrieval quality
- reduce semantic fragmentation

---

## 2. Embeddings

Uses dense vector embeddings to convert text into semantic representations.

Model used:

```python
BAAI/bge-small-en-v1.5
```

---

## 3. Vector Search

Uses FAISS for fast nearest-neighbor similarity search.

Implemented with:

```python
faiss.IndexFlatIP
```

---

## 4. Hybrid Retrieval

Retrieval combines:

- semantic similarity
- keyword boosting
- evidence scoring
- reranking

This improves answer relevance significantly over pure vector search.

---

## 5. Query Expansion

The system enriches user queries with additional evidence terms to improve recall.

Example:

```python
expanded_query = query + " additional evidence terms..."
```

---

## 6. Prompt Engineering

Retrieved context is transformed into structured prompts before generation.

The prompts:
- reduce hallucinations
- avoid direct copying
- focus answers on evidence

---

# 🤖 Custom GPT Implementation

This project includes a custom GPT-style Transformer implementation from scratch.

Implemented components:

- Token embeddings
- Positional embeddings
- Layer normalization
- GELU activation
- Feed-forward networks
- Multi-head self-attention
- Residual connections
- Causal masking
- Transformer blocks
- Autoregressive generation

---

# 🔥 Multi-Head Attention

The custom Transformer implements causal self-attention using masking.

```python
torch.triu(...)
```

This prevents future token leakage during generation.

---

# 🎲 Text Generation

The custom generator supports:

- Temperature scaling
- Top-k sampling
- Probabilistic token generation

This mimics modern autoregressive LLM inference.

---

# 🧠 GPT-2 Weight Loading

The project downloads and loads GPT-2 weights dynamically.

Features:
- TensorFlow checkpoint loading
- Parameter mapping into PyTorch
- Weight assignment into custom architecture

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>

cd custom-rag-system
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Install Ollama

This project uses a locally hosted Mistral model through Ollama.

Download Ollama:

https://ollama.com

---

# ⬇️ Download Mistral Model

After installing Ollama:

```bash
ollama pull mistral
```

---

# ▶️ Start Ollama Server

Before running the project:

```bash
ollama serve
```

---

# 🚀 Running the Project

# Mode 1 — Mistral Mode

Uses:
- Ollama
- local Mistral model
- production-style inference

Run:

```bash
python main.py --mode mistral
```

---

# Mode 2 — Custom GPT Mode

Uses:
- custom Transformer implementation
- GPT-2 weight loading
- autoregressive generation

Run:

```bash
python main.py --mode custom
```

---

# 💬 Example Workflow

```text
Ask a question:
Why did Gisburn stop painting?

↓
System retrieves relevant chunks

↓
Hybrid retrieval reranks evidence

↓
Prompt constructed

↓
Mistral or Custom GPT generates answer
```

---

# 📖 Example Retrieval Output

```text
--- Retrieved Context ---

--- Chunk 1 ---
...
```

The system shows retrieved evidence before generation for transparency and debugging.

---

# 🎯 Learning Outcomes

This project helped build understanding of:

## LLM Engineering
- Transformer internals
- Attention mechanisms
- Token generation

## RAG Engineering
- Embeddings
- Retrieval systems
- Vector databases
- Reranking

## Deep Learning
- Neural architectures
- Weight loading
- Sampling strategies

## Information Retrieval
- Query expansion
- Hybrid search
- Semantic similarity

## AI Infrastructure
- Ollama
- Local inference
- FAISS indexing

---

# 📌 Why This Project Matters

Most beginner AI projects only use API wrappers.

This project instead implements:
- custom Transformer internals
- hybrid retrieval logic
- vector search systems
- autoregressive generation
- local LLM orchestration

This provides a deeper understanding of how modern AI systems actually work internally.

---

# ⚠️ Notes

- Ollama must be installed for Mistral mode.
- The Ollama server must be running before execution.
- GPT-2 weights are downloaded automatically during first run.
- Large model files are excluded from GitHub.

---

# 🛠️ Future Improvements

Potential upgrades:

- Streamlit UI
- FastAPI backend
- Agentic RAG
- RAGAS evaluation
- Reranker models
- Hybrid BM25 retrieval
- Multi-document ingestion
- PDF parsing
- Conversational memory
- Real-time streaming

---

# 👨‍💻 Author

Built as a deep-learning and LLM engineering project to understand:
- how GPT models work internally
- how RAG systems retrieve knowledge
- how modern AI pipelines combine retrieval and generation

---
