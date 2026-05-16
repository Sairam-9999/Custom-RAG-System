# Adaptive Retrieval Intelligence Platform

### Retrieval-Oriented LLM Orchestration, Grounded Generation, and Retrieval Evaluation Research System

A research-focused adaptive Retrieval-Augmented Generation (RAG) platform exploring:

* adaptive retrieval orchestration
* grounded generation
* retrieval intelligence
* reasoning-aware context compression
* retrieval evaluation and observability
* hallucination reduction
* self-correcting retrieval systems
* synthetic dataset generation
* retrieval-aware fine-tuning
* local LLM inference
* future retrieval-native reasoning architectures

This project evolved from a simple semantic-search RAG prototype into a modern retrieval intelligence research platform focused on building production-oriented grounded AI systems capable of handling complex reasoning and domain-agnostic question answering.

Model weights are not committed. Run the training/download scripts to recreate them locally.

---

# 🧠 Core Research Direction

The primary focus of this project is no longer “prompt engineering” or “simple vector search.”

The system is evolving toward:

```text
Retrieval-Native Intelligence
```

where retrieval itself becomes intelligent through:

* adaptive query understanding
* retrieval planning
* reranking orchestration
* reasoning-aware evidence selection
* grounding validation
* retrieval evaluation
* retry/self-correction loops
* evidence-oriented orchestration

The long-term goal is to build a generalized retrieval system capable of supporting:

* factual QA
* reasoning-heavy queries
* comparative analysis
* temporal understanding
* multi-hop retrieval
* grounded conversational systems
* future custom retrieval-oriented GPT systems

---

# 🚀 What This Project Does

The system:

1. Reads `.txt` documents
2. Splits documents into semantic chunks
3. Generates embeddings using Sentence Transformers
4. Stores embeddings inside FAISS vector indexes
5. Performs hybrid retrieval:

   * dense retrieval
   * BM25 retrieval
6. Dynamically adapts retrieval policies based on query type
7. Applies reranking and evidence-aware context compression
8. Generates grounded answers using:

   * Mistral (via Ollama)
   * Fine-Tuned GPT-2
9. Validates groundedness and hallucination risk
10. Supports retrieval retries and self-correction loops
11. Tracks retrieval metrics and observability diagnostics

The system also supports grounded refusals:

```text
I don't know from the provided context.
```

---

# 🧠 Current Adaptive Retrieval Architecture

```text
User Query
    ↓
Query Understanding
    ↓
Adaptive Retrieval Policy
    ↓
Hybrid Retrieval
(Dense + BM25)
    ↓
RetrievalResult Objects
    ↓
CrossEncoder Reranking
    ↓
Reasoning-Aware Context Selection
    ↓
Compressed Evidence Context
    ↓
Generator
   ├── Mistral
   └── Fine-Tuned GPT-2
    ↓
Grounding Validation
    ↓
Retry / Self-Correction
    ↓
Final Grounded Response
    ↓
Evaluation + Observability
```

---

# 📂 Current Project Structure

```text
Custom-RAG-System/
├── assets/
│   ├── Unseen result 2.png
│   ├── Unseen result 3.png
│   ├── Unseen result 4.png
│   ├── Unseen result 5.png
│   ├── Unseen result.png
│   ├── ss.png
│   └── ssc.png
├── cache/
│   └── indexes/
├── data/
├── model/
│   ├── custom_gpt_updated.py
│   ├── fine_tune_gpt2_rag.py
│   ├── generate_large_rag_dataset.py
│   ├── gpt_download_updated.py
│   ├── rag_eval_mixed_large.jsonl
│   ├── rag_mixed_dataset_manifest.json
│   ├── rag_train_mixed_large.jsonl
│   ├── rag_training_mixed_large_all.jsonl
│   └── test_finetuned_gpt2.py
├── rag/
│   ├── __init__.py
│   ├── answering/
│   │   ├── __init__.py
│   │   ├── evidence_answer_extractor.py
│   │   ├── extractive_fallback.py
│   │   ├── generator_finetuned.py
│   │   ├── generator_mistral.py
│   │   ├── prompt_orchestrator.py
│   │   └── validator.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── types.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── eval_types.py
│   │   ├── evaluator.py
│   │   └── retrieval_metrics.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── file_fingerprint.py
│   │   └── vector_store.py
│   ├── pipeline.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── decomposition.py
│   │   ├── parser.py
│   │   ├── policy_engine.py
│   │   └── slot_extractor.py
│   └── retrieval/
│       ├── __init__.py
│       ├── context_selector.py
│       ├── reranker.py
│       ├── retriever.py
│       └── retry.py
└── main.py
```

---

# 🔥 Modern Retrieval Intelligence Features

## ✅ Adaptive Query Understanding

Supports:

* query classification
* reasoning query detection
* comparative query handling
* temporal query handling
* procedural query handling
* analytical query handling
* multi-hop query handling
* decomposition planning
* slot extraction
* semantic answer targeting

---

## ✅ Hybrid Retrieval

Combines:

* semantic vector retrieval
* BM25 lexical retrieval
* hybrid evidence scoring
* deduplication
* adaptive candidate selection

---

## ✅ CrossEncoder Reranking

Pipeline:

```text
Hybrid Retrieval
      ↓
CrossEncoder Reranker
      ↓
Top Evidence Chunks
```

Supports:

```text
BAAI/bge-reranker-base
```

Improves:

* retrieval precision
* evidence ordering
* noisy-context robustness
* grounded answer quality

---

## ✅ Reasoning-Aware Context Compression

Context selector performs:

* sentence extraction
* evidence scoring
* redundancy filtering
* token budgeting
* compressed evidence generation

Pipeline:

```text
Reranked Chunks
      ↓
Context Selector
      ↓
Compressed Evidence
      ↓
LLM
```

Improves:

* grounding density
* token efficiency
* hallucination resistance
* retrieval focus

---

## ✅ Grounding Validation

Validation layer supports:

* hallucination detection
* unsupported claim detection
* evidence coverage scoring
* contradiction detection
* confidence scoring
* grounded refusal handling

---

## ✅ Retrieval Retry + Self-Correction

The system supports validation-driven retrieval retries:

```text
Retrieve
    ↓
Generate
    ↓
Validate
    ↓
Retry Retrieval
    ↓
Regenerate
```

Supports:

* query reformulation
* retrieval expansion
* rerank depth increases
* context broadening
* query decomposition retries

---

## ✅ Retrieval Evaluation Framework

The project now includes a dedicated retrieval evaluation layer.

Supports:

* precision@k
* recall@k
* hit@k
* MRR
* evidence coverage
* stage-wise retrieval evaluation
* reranker evaluation
* context selector evaluation

This transformed the project from:

```text
RAG experimentation
```

into:

```text
Retrieval observability and evaluation research
```

---

# 🧪 Example Query Types

## Factual

```text
What destroyed kingdoms faster than swords?
```

## Reasoning

```text
Why did fleets continue deeper into the darkness despite fear?
```

## Comparative

```text
Compare the Titan Republic and the Nyx Syndicate.
```

## Temporal

```text
What happened during the age of expansion?
```

## Multi-Hop

```text
Which characters remembered the Astral Monks' prophecies and what did they warn?
```

---

# 🧠 Fine-Tuned GPT-2 Research

Supports:

* GPT-2 124M
* GPT-2 355M

Training pipeline includes:

* retrieval-grounded QA
* synthetic instruction generation
* refusal behavior
* noisy retrieval simulation
* distractor rejection
* hallucination reduction

---

# 🔥 Synthetic Dataset Generation

The system includes a fully generic synthetic dataset generator.

Pipeline:

```text
Documents
    ↓
Automatic QA Generation
    ↓
Question Augmentation
    ↓
Noisy Retrieval Simulation
    ↓
Refusal Example Injection
    ↓
Train / Eval Split
    ↓
Fine-Tuning
```

Supports:

* factual QA
* reasoning QA
* refusal behavior
* distractor examples
* noisy retrieval simulations
* paraphrased questions

---

# ⚡ Local Mistral Inference

Supports local inference through Ollama:

```powershell
ollama pull mistral
ollama run mistral
```

Mistral provides significantly stronger reasoning and grounding than smaller GPT-2 models.

---

# 📈 Retrieval Observability

Tracks:

```text
retrieval_ms
rerank_ms
compression_ms
generation_ms
compression_ratio
refused
evidence_coverage
hallucination_flags
```

The system now supports measurable retrieval diagnostics instead of blind optimization.

---

# 🛠️ Installation

## 1. Create Environment

```powershell
conda create -n llm-rag python=3.10 -y
conda activate llm-rag
```

---

## 2. Install Dependencies

```powershell
pip install torch transformers sentence-transformers faiss-cpu numpy pandas tiktoken requests
```

---

## 3. Install Ollama

Install:

```text
https://ollama.com
```

Then run:

```powershell
ollama pull mistral
ollama run mistral
```

---

# 🚀 Running the System

## Mistral Mode

```powershell
python main.py --mode mistral --file data/The_Verdict.txt
```

---

## Fine-Tuned GPT-2 Mode

```powershell
python main.py --mode finetuned --file data/The_Verdict.txt
```

---

# 🧠 Generate Synthetic Dataset

```powershell
python model/generate_large_rag_dataset.py
```

---

# 🔥 Fine-Tune GPT-2

```powershell
python model/fine_tune_gpt2_rag.py --train model/rag_train_mixed_large.jsonl --eval model/rag_eval_mixed_large.jsonl --model-size 355M --epochs 1 --max-length 256
```

---

# 🧪 Test Fine-Tuned Model

```powershell
python model/test_finetuned_gpt2.py
```

---

# 🧠 Major Engineering Learnings

## Retrieval Quality Dominates RAG Quality

As the system evolved, retrieval orchestration became more important than model size alone.

Improving:

* retrieval precision
* reranking
* context compression
* evidence quality

often improved grounded answers more than fine-tuning itself.

---

## Small GPT Models Overfit Easily

Smaller GPT-2 models:

* hallucinate confidently
* memorize patterns
* struggle on unseen contexts
* require strong retrieval grounding

---

## Grounded QA Requires Specialized Training

The model must explicitly learn:

* answer extraction
* grounding behavior
* refusal handling
* distractor rejection
* evidence focus

---

## Synthetic Data Is Extremely Powerful

Synthetic QA generation enabled scalable grounded dataset creation for retrieval-aware training.

---

# ⚠️ Current Research Bottlenecks

Current limitations include:

* reasoning-aware evidence compression
* bridge fact preservation
* multi-hop evidence orchestration
* contradiction resolution
* long-context grounding
* retrieval observability at scale
* reasoning chain construction

---

# 🚀 Future Research Direction

Ongoing exploration includes:

* evidence graph construction
* reasoning-aware retrieval
* bridge fact preservation
* retrieval-native reasoning
* source reliability modeling
* memory-aware retrieval
* long-context orchestration
* agentic retrieval planning
* contradiction resolution
* conversational RAG
* retrieval-aware instruction tuning
* adaptive evidence orchestration

---

# 🧠 Future Custom GPT Direction

The long-term goal is evolving toward custom retrieval-oriented GPT systems capable of:

* reasoning over retrieved evidence
* preserving grounding
* reducing hallucinations
* adaptive retrieval planning
* multi-step evidence orchestration
* memory-aware retrieval
* domain-agnostic grounded QA
* retrieval-native reasoning

---

# 🏁 Final Goal

The long-term goal is building:

```text
Production-grade Retrieval Intelligence Systems
```

with:

* adaptive retrieval orchestration
* grounded generation
* reasoning-aware retrieval
* advanced reranking
* retrieval evaluation
* observability
* self-correcting retrieval loops
* retrieval-native reasoning
* custom retrieval-oriented GPT systems
* scalable grounded AI architectures
