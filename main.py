import argparse

from rag.rag_pipeline import build_rag, ask_rag
from rag.generator import generate_answer as custom_generate
from rag.generator_mistral import generate_answer as mistral_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["custom", "mistral"],
        default="mistral",
    )
    args = parser.parse_args()

    print(f"\nRunning in mode: {args.mode.upper()}\n")

    store, chunks = build_rag("data/big.txt")
    query = input("Ask a question: ")

    evidence_prompt, mistral_prompt, context_chunks = ask_rag(query, store, chunks)

    print("\n--- Retrieved Context ---\n")
    for i, c in enumerate(context_chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(c)
        print("--- END CHUNK ---")

    print("\n--- Generating Answer ---\n")

    if args.mode == "custom":
        final_prompt = evidence_prompt + "\nAnswer:"
        answer = custom_generate(final_prompt)
    elif args.mode == "mistral":
        answer = mistral_generate(mistral_prompt)

    print(answer)


if __name__ == "__main__":
    main()
