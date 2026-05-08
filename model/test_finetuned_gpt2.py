import argparse
from pathlib import Path
import sys
import torch
import tiktoken


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from custom_gpt_updated import GPTModel, MODEL_CONFIGS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EOS_TOKEN = "<|endoftext|>"
TOKENIZER = tiktoken.get_encoding("gpt2")
EOS_ID = TOKENIZER.encode(EOS_TOKEN, allowed_special={EOS_TOKEN})[0]


def load_model(checkpoint_path: Path, model_size: str):
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_CONFIGS.keys())}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = MODEL_CONFIGS[model_size]
    model = GPTModel(config).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config


def build_prompt(context: str, question: str):
    return f"""### Instruction:
Answer using only the provided context.
Return the shortest correct answer.
Do not use outside knowledge.
If the answer is not in the context, say: I don't know from the provided context.

### Context:
{context}

### Question:
{question}

### Answer:
"""


def clean_answer(text: str):
    text = text.replace(EOS_TOKEN, "").strip()
    return " ".join(text.split())


@torch.no_grad()
def generate(model, config, prompt, max_new_tokens=30, temperature=0.1):
    input_ids = TOKENIZER.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = input_tensor[:, -config["context_length"]:]
        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Greedy decoding for factual QA
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.item() == EOS_ID:
            break

        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    output_ids = input_tensor[0].tolist()
    answer_ids = output_ids[len(input_ids):]
    return clean_answer(TOKENIZER.decode(answer_ids))


def load_context_from_file(path: Path, max_chars: int = 2500):
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text[:max_chars]


def main():
    parser = argparse.ArgumentParser(description="Generic fine-tuned GPT-2 RAG checkpoint tester")

    parser.add_argument(
        "--checkpoint",
        default=str(BASE_DIR / "rag_finetuned_gpt2_ckpt" / "model_final.pt"),
        help="Path to fine-tuned checkpoint",
    )

    parser.add_argument(
        "--model-size",
        default="355M",
        choices=list(MODEL_CONFIGS.keys()),
        help="GPT-2 model size used during training",
    )

    parser.add_argument(
        "--context-file",
        default=None,
        help="Optional text file to use as context",
    )

    parser.add_argument(
        "--context",
        default=None,
        help="Optional raw context string",
    )

    parser.add_argument(
        "--question",
        default=None,
        help="Single question to test",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum answer tokens",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Low temperature recommended for factual QA",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Model size: {args.model_size}")

    model, config = load_model(checkpoint_path, args.model_size)

    if args.context:
        context = args.context
    elif args.context_file:
        context_path = Path(args.context_file)
        if not context_path.is_absolute():
            context_path = PROJECT_ROOT / context_path
        context = load_context_from_file(context_path)
    else:
        context = """STORY 2: THE RED-HEADED LEAGUE
Date: April 27, 1890 → October 9, 1890
Entities: Holmes, Watson, Jabez Wilson, Spaulding
Money: $4/week
Events:
- Fake job created
- Wilson distracted
- Tunnel dug to bank
Outcome: Crime stopped
Insight: Distraction hides real objective"""

    if args.question:
        questions = [args.question]
    else:
        questions = [
            "What is the date range for The Red-Headed League?",
            "What insight does The Red-Headed League show?",
            "What is Watson's age?",
        ]

    for question in questions:
        prompt = build_prompt(context, question)
        answer = generate(
            model=model,
            config=config,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print("\nQUESTION:", question)
        print("ANSWER:", answer)


if __name__ == "__main__":
    main()