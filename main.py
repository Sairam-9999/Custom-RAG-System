import argparse
from pathlib import Path

from rag.rag_pipeline import build_rag, ask_rag
from rag.generator_mistral import generate_answer as mistral_generate
from rag.generator_finetuned import generate_answer as finetuned_generate, load_finetuned_model


def print_context(context_chunks, top_n=3):
    print("\n--- Retrieved Context ---\n")

    for i, chunk in enumerate(context_chunks[:top_n], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
        print("--- END CHUNK ---")


def main():
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
        default="data/magic_academy_100kb_story.txt",
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

    args = parser.parse_args()

    data_path = Path(args.file)
    if not data_path.exists():
        raise FileNotFoundError(f"TXT file not found: {data_path}")

    print(f"\nRunning in mode: {args.mode.upper()}")
    print(f"Using data file: {data_path}\n")

    # 1. Build RAG index from selected TXT file
    store, chunks = build_rag(str(data_path))

    # 2. Load model once if using finetuned mode
    finetuned_model = None
    if args.mode == "finetuned":
        print("Loading fine-tuned GPT-2 model...")
        finetuned_model = load_finetuned_model(checkpoint_path=args.checkpoint)
        print("Model loaded.\n")

    # 3. Ask user question
    query = input("Ask a question: ")

    # 4. Retrieve context + build prompts
    mistral_prompt, context_chunks = ask_rag(query, store, chunks)

    if not args.hide_context:
        print_context(context_chunks, top_n=10)

    print("\n--- Generating Answer ---\n")

    # 5. Generate final answer
    if args.mode == "mistral":
        answer = mistral_generate(mistral_prompt)

    elif args.mode == "finetuned":
        context = "\n\n".join(context_chunks[:2])

        answer = finetuned_generate(
            context=context,
            question=query,
            model=finetuned_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print(answer)


if __name__ == "__main__":
    main()
