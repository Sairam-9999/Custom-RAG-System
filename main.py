import argparse
from pathlib import Path

import time

from rag.pipeline import build_rag, ask_rag
from rag.retrieval import CrossEncoderReranker, ContextSelector
from rag.answering.generator_mistral import generate_answer as mistral_generate


def print_context(context_chunks, top_n=3):
    print("\n--- Retrieved Context ---\n")

    for i, result in enumerate(context_chunks[:top_n], start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"chunk_id:        {result.chunk_id}")
        print(f"semantic_score:  {result.semantic_score:.4f}")
        print(f"bm25_score:      {result.bm25_score:.4f}")
        print(f"hybrid_score:    {result.hybrid_score:.4f}")
        print(f"rerank_score:    {result.rerank_score if result.rerank_score is not None else None}")
        print()
        print(result.text)
        print("--- END CHUNK ---")


def print_evidence(compressed, results_map):
    print("\n--- Selected Evidence Sentences ---\n")

    for sentence in compressed.selected_sentences:
        # Find the originating chunk for score provenance
        origin = None
        for r in results_map:
            if sentence in r.text:
                origin = r
                break

        if origin:
            print(f"[Chunk {origin.chunk_id}]")
            print(f"Rerank Score:  {origin.rerank_score if origin.rerank_score is not None else 'N/A'}")
            print()
        print(f'"{sentence}"')
        print()

    if compressed.metadata:
        print("Selector metadata:")
        for k, v in compressed.metadata.items():
            print(f"  {k}: {v}")
    print("--- END EVIDENCE ---")


def main():
    # START TOTAL EXECUTION TIMER
    total_start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Local RAG runner using Mistral or fine-tuned GPT-2."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["mistral", "finetuned"],
        default="mistral",
        help="Generator backend to use.",
    )

    parser.add_argument(
        "--file",
        type=str,
        default="data/space_epic_story_600kb.txt",
        help="TXT file to index and test with RAG.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/rag_finetuned_gpt2_ckpt/model_final.pt",
        help="Fine-tuned GPT-2 checkpoint path.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum new tokens for fine-tuned GPT-2 generation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Fine-tuned GPT-2 generation temperature.",
    )

    parser.add_argument(
        "--hide-context",
        action="store_true",
        help="Hide retrieved chunks from terminal output.",
    )

    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking after hybrid retrieval.",
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Cross-encoder model name for reranking.",
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=30,
        help="Number of candidates to retrieve before reranking.",
    )

    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="Number of chunks to keep after reranking.",
    )

    parser.add_argument(
        "--context-top-n",
        type=int,
        default=3,
        help="Number of top chunks to feed into the generator prompt.",
    )

    parser.add_argument(
        "--use-context-selector",
        action="store_true",
        help="Enable evidence-based context compression before generation.",
    )

    parser.add_argument(
        "--max-context-sentences",
        type=int,
        default=8,
        help="Maximum sentences the context selector may include.",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2500,
        help="Maximum characters the context selector may include.",
    )

    parser.add_argument(
        "--workload-type",
        type=str,
        choices=["fact", "analytical", "mixed"],
        default="mixed",
        help="Expected query workload for optimal chunking: 'fact' (small chunks), 'analytical' (medium chunks), 'mixed' (default).",
    )

    args = parser.parse_args()

    data_path = Path(args.file)
    if not data_path.exists():
        raise FileNotFoundError(f"TXT file not found: {data_path}")

    print(f"\nRunning in mode: {args.mode.upper()}")
    print(f"File: {args.file}")
    print(f"Query-driven orchestration: Enabled\n")

    # 1. Build RAG index from selected TXT file with workload-optimized chunking
    store, chunks = build_rag(str(data_path), workload_type=args.workload_type)

    # 2. Load model once if using finetuned mode
    finetuned_model = None
    if args.mode == "finetuned":
        from rag.answering.generator_finetuned import generate_answer as finetuned_generate, load_finetuned_model
        print("Loading fine-tuned GPT-2 model...")
        finetuned_model = load_finetuned_model(checkpoint_path=args.checkpoint)
        print("Model loaded.\n")

    # 3. Optionally load reranker
    reranker = None
    if args.use_reranker:
        print(f"Loading reranker: {args.reranker_model} ...")
        reranker = CrossEncoderReranker(model_name=args.reranker_model)
        print("Reranker loaded.\n")

    # 4. Optionally load context selector
    context_selector = None
    if args.use_context_selector:
        context_selector = ContextSelector(
            max_sentences=args.max_context_sentences,
            max_chars=args.max_context_chars,
        )
        print("Context selector enabled.\n")

    # 5. Ask user question
    query = input("Ask a question: ")

    # 7. Retrieve context + build prompts with query-driven orchestration
    answer_or_prompt, context_chunks, compressed, timings, analysis = ask_rag(
        query,
        store,
        chunks,
        reranker=reranker,
        context_selector=context_selector,
        retrieval_top_k=args.retrieval_top_k,
        rerank_top_n=args.rerank_top_n,
        context_top_n=args.context_top_n,
    )

    # If deterministic answer was extracted directly, print and exit
    if timings.get("answer_source") == "deterministic_extraction":
        print(answer_or_prompt)
        print("\n--- Execution Complete ---")
        return

    # Extract adaptive configuration for generation
    adaptive_config = timings.get("adaptive_config", {})
    max_new_tokens = adaptive_config.get("max_new_tokens", args.max_new_tokens)
    use_extractive_first = adaptive_config.get("use_extractive_first", False)

    if not args.hide_context:
        print_context(context_chunks, top_n=args.rerank_top_n if args.use_reranker else args.retrieval_top_k)
        if compressed is not None:
            print_evidence(compressed, context_chunks)

    print("\n--- Generating Answer ---\n")

    # 8. Generate final answer (timed) with adaptive parameters
    t_gen_start = time.perf_counter()
    if args.mode == "mistral":
        # Prepare context for fallback
        if compressed is not None:
            context_for_fallback = compressed.context
        else:
            context_for_fallback = "\n\n".join(r.text for r in context_chunks[:args.context_top_n])
        
        # Use adaptive max_new_tokens for concise factual answers
        # Pass query_analysis for query-aware fallback
        answer = mistral_generate(answer_or_prompt, num_predict=max_new_tokens, context=context_for_fallback, question=query, query_analysis=analysis)

    elif args.mode == "finetuned":
        if compressed is not None:
            context = compressed.context
        else:
            context = "\n\n".join(r.text for r in context_chunks[:args.context_top_n])

        # Use adaptive parameters: extractive shortcut for facts, optimized token limits
        answer = finetuned_generate(
            context=context,
            question=query,
            model=finetuned_model,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            use_extractive_first=use_extractive_first,
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    timings["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000
    timings["total_ms"] = sum(v for v in timings.values() if isinstance(v, (int, float)))

    print(answer)

    # TOTAL EXECUTION TIME
    total_execution_time = time.perf_counter() - total_start_time

    print("\n--- Execution Complete ---")
    print(f"Total execution time: {total_execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
