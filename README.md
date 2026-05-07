# 🧠 RAG + LLM System with Fine-Tuned GPT-2, Custom Transformer, and Mistral

A fully custom Retrieval-Augmented Generation (RAG) system built from scratch using:

* Fine-Tuned GPT-2
* Custom Transformer/GPT implementation
* FAISS vector search
* Hybrid retrieval
* Semantic embeddings
* Query expansion
* Local Mistral inference using Ollama
* Autoregressive text generation

This project demonstrates the internal mechanics behind modern LLM systems and production-style RAG pipelines.

---

# 🚀 Features

## 🔍 Retrieval-Augmented Generation (RAG)

* Document chunking with overlap
* Semantic embedding generation
* FAISS vector database
* Hybrid retrieval (semantic + keyword reranking)
* Query expansion
* Context-aware answer generation

---

## 🤖 Multiple Generation Modes

### 1. Mistral Mode (Production-style)

Uses:

* Ollama
* Local Mistral model
* REST API inference

---

### 2. Custom GPT Mode

Uses:

* Fully custom Transformer implementation
* Multi-head self-attention
* Positional embeddings
* GPT-2 weight loading
* Autoregressive generation

---

### 3. Fine-Tuned GPT-2 Mode

Uses:

* Fine-tuned GPT-2 checkpoints
* Retrieval-grounded QA generation
* Supervised fine-tuning pipeline
* Evaluation/testing workflow

---

# 🧠 What This Project Demonstrates

This project was built to deeply understand how modern LLM systems work internally.

Instead of only calling APIs, this system implements:

* Transformer architecture
* Attention mechanism
* Autoregressive generation
* Vector retrieval systems
* Embedding pipelines
* Hybrid search
* Prompt engineering
* Fine-tuning workflows
* Evaluation pipelines
* Local LLM orchestration

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

# 🔥 Fine-Tuning Pipeline

The project includes a supervised GPT-2 fine-tuning pipeline for retrieval-style question answering.

Pipeline:

```text
Training Dataset
        ↓
Tokenization
        ↓
GPT-2 Fine-Tuning
        ↓
Checkpoint Saving
        ↓
Evaluation
        ↓
Inference Testing
```

Fine-tuning helps the model learn:

* retrieval-grounded responses
* structured QA behavior
* improved contextual generation

---

# 📂 Updated Project Structure

```text
RAG plus LLM/
├── data/
│   ├── The_Verdict.txt
│   └── big.txt
│
│
├── main.py
│
├── model/
│   ├── custom_gpt_updated.py
│   ├── fine_tune_gpt2_rag.py
│   ├── gpt_download_updated.py
│   ├── rag_eval_sherlock.jsonl
│   ├── rag_train_sherlock.jsonl
│   ├── test_finetuned_gpt2.py
│
├── rag/
│   ├── __init__.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── generator_finetuned.py
│   ├── generator_mistral.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
│
├── assests/
│   ├── ss
│   ├── ssc
```

---

# ⚙️ Technologies Used

| Component         | Technology             |
| ----------------- | ---------------------- |
| Deep Learning     | PyTorch                |
| Embeddings        | Sentence Transformers  |
| Embedding Model   | BAAI/bge-small-en-v1.5 |
| Vector Database   | FAISS                  |
| Local LLM Runtime | Ollama                 |
| External Model    | Mistral                |
| Tokenization      | tiktoken               |
| GPT Weights       | GPT-2 124M             |
| Backend Logic     | Python                 |

---

# 🧩 Core Components

## 1. Chunking

Splits large documents into overlapping chunks for retrieval.

```python
chunk_size=900
overlap=250
```

Purpose:

* preserve context continuity
* improve retrieval quality
* reduce semantic fragmentation

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

* semantic similarity
* keyword boosting
* evidence scoring
* reranking

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

* reduce hallucinations
* avoid direct copying
* focus answers on evidence

---

# 🤖 Custom GPT Implementation

This project includes a custom GPT-style Transformer implementation from scratch.

Implemented components:

* Token embeddings
* Positional embeddings
* Layer normalization
* GELU activation
* Feed-forward networks
* Multi-head self-attention
* Residual connections
* Causal masking
* Transformer blocks
* Autoregressive generation

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

* Temperature scaling
* Top-k sampling
* Probabilistic token generation

This mimics modern autoregressive LLM inference.

---

# 🧠 GPT-2 Weight Loading

The project downloads and loads GPT-2 weights dynamically.

Features:

* TensorFlow checkpoint loading
* Parameter mapping into PyTorch
* Weight assignment into custom architecture

---

# 📊 Evaluation

The project includes evaluation datasets and testing pipelines.

Evaluation focuses on:

* retrieval quality
* grounded generation
* hallucination observation
* answer relevancy
* unseen document testing

The system also displays retrieved context before generation for transparency and debugging.

---

# 🚀 Running the Project

# Mode 1 — Mistral Mode

Uses:

* Ollama
* local Mistral model
* production-style inference

Run:

```bash
python main.py --mode mistral
```

---

# Mode 2 — Custom GPT Mode

Uses:

* custom Transformer implementation
* GPT-2 weight loading
* autoregressive generation

Run:

```bash
python main.py --mode custom
```

---

# Mode 3 — Fine-Tuned GPT-2 Mode

Uses:

* fine-tuned GPT-2 checkpoints
* retrieval-grounded generation
* trained QA behavior

Run:

```bash
python main.py --mode finetuned
```

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>

cd "RAG plus LLM"
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

# 💬 Example Workflow

```text
Ask a question
        ↓
System retrieves relevant chunks
        ↓
Hybrid retrieval reranks evidence
        ↓
Prompt constructed
        ↓
Mistral / Custom GPT / Fine-Tuned GPT generates answer
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

* Transformer internals
* Attention mechanisms
* Token generation
* GPT architectures

## RAG Engineering

* Embeddings
* Retrieval systems
* Vector databases
* Reranking pipelines

## Fine-Tuning

* Supervised training
* Checkpoint management
* Inference testing
* Retrieval-grounded QA

## Information Retrieval

* Query expansion
* Hybrid search
* Semantic similarity

## AI Infrastructure

* Ollama
* Local inference
* FAISS indexing

---

# 📌 Why This Project Matters

Most beginner AI projects only use API wrappers.

This project instead implements:

* custom Transformer internals
* hybrid retrieval logic
* vector search systems
* fine-tuning pipelines
* autoregressive generation
* local LLM orchestration

This provides a deeper understanding of how modern AI systems actually work internally.

---

# ⚠️ Current Limitations

* Small GPT-2 models can still hallucinate
* Fine-tuned models may struggle on unseen datasets
* Retrieval quality heavily affects answer quality
* No conversational memory yet
* No advanced reranker models yet

---

# 🛠️ Future Improvements

Potential upgrades:

* Streamlit UI
* FastAPI backend
* Agentic RAG
* RAGAS evaluation
* Cross-encoder rerankers
* Hybrid BM25 retrieval
* LoRA / QLoRA fine-tuning
* Multi-document ingestion
* PDF parsing
* Conversational memory
* Real-time streaming

---

# 🧠 In Simple Terms

## What I Built

I built a system that can:

* search documents,
* retrieve relevant information,
* and generate answers using multiple LLM approaches.

---

## Why I Built It

To deeply understand how modern AI systems combine:

* retrieval,
* transformers,
* embeddings,
* and text generation.

---

## How It Works

1. Documents are split into chunks
2. Chunks become embeddings
3. FAISS retrieves relevant information
4. Retrieved context is injected into prompts
5. GPT/Mistral generates grounded answers

---

# 📈 Fine-Tuning Observations

During experimentation:

| Observation                                        | Insight                           |
| -------------------------------------------------- | --------------------------------- |
| Fine-tuned GPT performs better on trained-style QA | Model learns dataset patterns     |
| Performance drops on unseen files                  | Generalization limitations exist  |
| Retrieval quality impacts generation quality       | Better retrieval improves answers |
| Small GPT-2 models still hallucinate               | Model scale matters               |

---

# 👨‍💻 Author

Built as a deep-learning and LLM engineering project to understand:

* how GPT models work internally
* how RAG systems retrieve knowledge
* how modern AI pipelines combine retrieval and generation
