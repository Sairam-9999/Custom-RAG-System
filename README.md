# RAG + Fine-Tuned GPT-2 + Mistral Research System

A research-focused Retrieval-Augmented Generation (RAG) system exploring:

* semantic retrieval
* grounded QA
* synthetic dataset generation
* GPT-2 fine-tuning
* hallucination reduction
* realistic noisy-context training
* local Mistral inference
* retrieval-aware generation

This project evolved from a simple RAG prototype into a deeper exploration of modern retrieval-grounded LLM system design.
Model weights are not committed. Run the training/download scripts to recreate them locally.

---

# 🚀 What This Project Does

The system:

1. Reads `.txt` documents
2. Splits them into semantic chunks
3. Embeds chunks using Sentence Transformers
4. Stores embeddings in a vector index
5. Retrieves relevant chunks for a query
6. Generates answers using:

   * Mistral (via Ollama)
   * Fine-tuned GPT-2
7. Supports grounded refusals:

```text
I don't know from the provided context.
```

---

# 🧠 System Architecture

```text
User Question
      ↓
Retriever
      ↓
Top-K Relevant Chunks
      ↓
Generator
   ├── Mistral
   └── Fine-Tuned GPT-2
      ↓
Grounded Answer
```

---

# 📂 Current Project Structure

```text
RAG pus LLM/
├── data/
│   ├── The_Verdict.txt
│   ├── adventure_party_treasure_hunt_100kb_story.txt
│   ├── big.txt
│   ├── dragons_and_ancient_powers_100kb_story.txt
│   ├── epic_kingdoms_and_wars_100kb_story.txt
│   └── mythology_inspired_100kb_story.txt
│
├── model/
│   ├── custom_gpt_updated.py
│   ├── fine_tune_gpt2_rag.py
│   ├── generate_large_rag_dataset.py
│   ├── gpt_download_updated.py
│   ├── rag_train_mixed_large.jsonl
│   ├── rag_eval_mixed_large.jsonl
│   ├── rag_mixed_dataset_manifest.json
│   ├── test_finetuned_gpt2.py
│
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── generator_finetuned.py
│   ├── generator_mistral.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
│
└── main.py
```

---

# ⚡ Features

## ✅ Semantic Retrieval

Uses:

* Sentence Transformers
* vector similarity search
* evidence-aware reranking
* chunk deduplication

---

## ✅ Fine-Tuned GPT-2

Supports:

* GPT-2 124M
* GPT-2 355M

Custom training pipeline includes:

* retrieval-grounded QA
* refusal behavior
* noisy-context learning
* synthetic instruction generation

---

## ✅ Local Mistral Inference

Supports local inference through Ollama:

```powershell
ollama run mistral
```

This provides significantly stronger reasoning and grounding than small GPT-2 models.

---

## ✅ Synthetic Dataset Generation

Automatically generates:

* what questions
* who questions
* where questions
* when questions
* why questions
* how questions
* which questions
* did/was questions
* refusal examples

---

## ✅ Realistic RAG Training

The system trains using noisy multi-chunk retrieval simulation:

```text
Chunk 1 → distractor
Chunk 2 → answer
Chunk 3 → irrelevant
```

This teaches:

* evidence grounding
* distractor rejection
* hallucination reduction
* realistic retrieval behavior

---

# 🔥 How Dataset Generation Works

The project includes a fully generic synthetic dataset generator:

```text
Documents
    ↓
Automatic QA Generation
    ↓
Question Augmentation
    ↓
Noisy Chunk Construction
    ↓
Refusal Example Injection
    ↓
Train/Eval Split
    ↓
Fine-Tuning
```

The generator creates:

* grounded QA examples
* reasoning examples
* refusal behavior
* paraphrased questions
* noisy retrieval simulations

---

# 🧪 Example Questions

## Sherlock / Literary QA

```text
Who became the man of the moment?
What insight does The Red-Headed League show?
Who gave Gisburn the donkey sketch?
```

---

## Fantasy QA

```text
What destroyed kingdoms faster than swords?
Where did priests wearing bone masks chant?
Who controlled the black towers beneath the crimson moon?
```

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

https://ollama.com

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

Outputs:

```text
rag_train_mixed_large.jsonl
rag_eval_mixed_large.jsonl
rag_training_mixed_large_all.jsonl
```

---

# 🔥 Fine-Tune GPT-2

## GPT-2 355M

```powershell
python model/fine_tune_gpt2_rag.py --train model/rag_train_mixed_large.jsonl --eval model/rag_eval_mixed_large.jsonl --model-size 355M --epochs 1 --max-length 256
```

---

# 🧪 Test Fine-Tuned Model

```powershell
python model/test_finetuned_gpt2.py
```

Test on custom files:

```powershell
python model/test_finetuned_gpt2.py --context-file data/magic_academy_100kb_story.txt --question "Where did priests wearing bone masks chant beside rivers of fire?"
```

---

# 🧠 Major Engineering Learnings

## Retrieval Quality Dominates RAG Quality

Improving retrieval often improved answers more than fine-tuning alone.

---

## Small GPT Models Overfit Easily

Small GPT-2 models:

* memorize patterns quickly
* hallucinate confidently
* struggle on unseen contexts
* require strong grounding supervision

---

## Grounded QA Needs Specialized Training

Standard next-token prediction is not enough.

The model must explicitly learn:

* answer extraction
* evidence grounding
* refusal behavior
* distractor rejection

---

## Synthetic Data Is Extremely Powerful

The project demonstrates how synthetic QA generation can bootstrap large grounded datasets automatically.

This mirrors real-world modern LLM training strategies.

---

# 🚀 Recent Major Upgrades

## 🔥 Fully Generic Dataset Generation

The project evolved from handcrafted story-specific QA generation into a fully generic synthetic RAG dataset pipeline.

Supports arbitrary `.txt` documents.

---

## 🔥 Multi-Domain Training

Training now includes:

* mystery stories
* fantasy worlds
* mythology-inspired stories
* literary prose
* adventure stories
* war narratives

This improves:

* generalization
* retrieval robustness
* unseen-domain testing

---

## 🔥 Noisy Retrieval Training

Training contexts now simulate real RAG retrieval:

```text
Chunk 1 → irrelevant
Chunk 2 → answer
Chunk 3 → distractor
```

This teaches the model to focus on evidence-bearing chunks.

---

## 🔥 Grounded Refusal Training

The model now learns grounded refusals:

```text
I don't know from the provided context.
```

This reduces hallucinations on unsupported questions.

---

## 🔥 Improved Decoding

Generation improvements include:

* greedy decoding
* low-temperature factual inference
* EOS stopping
* repetition cleanup
* fallback refusal handling

---

## 🔥 Retrieval Improvements

Retriever upgrades include:

* semantic reranking
* evidence scoring
* query expansion
* overlap scoring
* chunk deduplication

---

## 🔥 GPT-2 Scaling Experiments

The project evolved from GPT-2 124M experiments into larger GPT-2 355M fine-tuning experiments.

The 355M model was successfully:

* downloaded locally
* loaded into the custom GPT architecture
* fine-tuned on synthetic RAG datasets
* evaluated on unseen retrieval contexts
* tested against multiple document domains

Research observations:

| Model            | Observation                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| GPT-2 124M       | weak grounding and unstable extraction                                       |
| GPT-2 355M       | noticeably stronger retrieval-grounded QA                                    |
| Fine-tuned GPT-2 | improved factual extraction but still struggles on difficult unseen contexts |
| Mistral          | significantly stronger reasoning and grounding                               |

The experiments demonstrated how model scale directly impacts:

* grounding quality
* answer extraction
* hallucination behavior
* retrieval robustness
* unseen-domain generalization

---

# 📈 Current Research Direction

Ongoing exploration includes:

* larger GPT-2 variants
* LoRA / QLoRA fine-tuning
* Mistral/Qwen instruction tuning
* hybrid BM25 + vector retrieval
* cross-encoder rerankers
* conversational RAG
* RAGAS evaluation
* long-context optimization
* agentic RAG systems

---

# ⚠️ Current Limitations

Small GPT-2 models still struggle with:

* difficult unseen contexts
* multi-hop reasoning
* exact answer extraction
* long-context grounding
* subtle entity disambiguation

Mistral performs significantly better for real-world QA quality.

---

# 🏁 Final Goal

The long-term goal is to evolve this into:

```text
Production-grade retrieval-grounded LLM system
```

with:

* scalable retrieval
* grounded generation
* advanced reranking
* robust evaluation
* synthetic instruction tuning
* agentic reasoning
* memory-aware RAG
* multi-model orchestration
