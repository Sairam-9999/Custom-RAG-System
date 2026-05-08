"""Fine-tune your local GPT-2-style model for RAG answering.

What this trains:
- Given: Instruction + Context + Question
- Predict: only the Answer tokens

Run:
    pip install torch tensorflow numpy requests tqdm tiktoken
    python fine_tune_gpt2_rag.py --train rag_train_sherlock.jsonl --eval rag_eval_sherlock.jsonl

Optional first-time GPT-2 download happens automatically into ./gpt2/124M.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken

from custom_gpt_updated import GPT_CONFIG_124M, GPTModel, load_weights_into_gpt, generate_text_simple
from gpt_download_updated import download_and_load_gpt2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "rag_train_mixed_large.jsonl"
EVAL_PATH = BASE_DIR / "rag_eval_mixed_large.jsonl"

MODEL_CONFIGS = {
    "124M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "355M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
}


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RAGInstructionDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int = 512):
        self.rows = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]

        prompt = ex["prompt"]
        completion = ex["completion"]

        eos_token = ""
        if not completion.endswith(eos_token):
            completion = completion + eos_token

        prompt_ids = self.tokenizer.encode(
            prompt,
            allowed_special={"<|endoftext|>"}
        )

        full_ids = self.tokenizer.encode(
            prompt + completion,
            allowed_special={"<|endoftext|>"}
        )

        full_ids = full_ids[: self.max_length]

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        prompt_len = min(len(prompt_ids), len(labels))
        labels = labels.copy()

        for i in range(max(prompt_len - 1, 0)):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, labels = [], []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=50256))
        labels.append(F.pad(item["labels"], (0, pad_len), value=-100))

    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}


def compute_loss(model: GPTModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    logits = model(input_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model: GPTModel, loader: DataLoader, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        losses.append(compute_loss(model, batch).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def build_test_prompt() -> str:
    return (
        "### Instruction:\n"
        "Answer the question using only the provided context. "
        "If the answer is not in the context, say: I don't know from the provided context.\n\n"
        "### Context:\n"
        "STORY 2: THE RED-HEADED LEAGUE\n"
        "Date: April 27, 1890 → October 9, 1890\n"
        "Money: $4/week\n"
        "Events:\n- Fake job created\n- Wilson distracted\n- Tunnel dug to bank\n\n"
        "### Question:\n"
        "What is the date range for The Red-Headed League case?\n\n"
        "### Answer:\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(TRAIN_PATH))
    parser.add_argument("--eval", default=str(EVAL_PATH))
    parser.add_argument("--model-size", default="124M", choices=["124M", "355M", "774M", "1558M"])
    parser.add_argument("--models-dir", default=str(BASE_DIR / "gpt2"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "rag_finetuned_gpt2_ckpt"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)  # FIX 5: Reduce epochs to prevent overfitting on tiny dataset
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means train full epochs")
    parser.add_argument("--skip-pretrained", action="store_true", help="debug only: do not load GPT-2 weights")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    tokenizer = tiktoken.get_encoding("gpt2")

    train_ds = RAGInstructionDataset(args.train, tokenizer, max_length=args.max_length)
    eval_ds = RAGInstructionDataset(args.eval, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {len(eval_ds)}")

    config = MODEL_CONFIGS[args.model_size]
    model = GPTModel(config).to(DEVICE)

    if not args.skip_pretrained:
        print("Downloading/loading GPT-2 pretrained weights...")
        settings, params = download_and_load_gpt2(args.model_size, args.models_dir)
        print(f"Loaded GPT-2 settings: n_layer={settings['n_layer']}, n_embd={settings['n_embd']}")
        load_weights_into_gpt(model, params)
        print("Pretrained GPT-2 weights copied into local model.")
    else:
        print("WARNING: skipping pretrained weights. This is only useful for debugging the loop.")

    # Fine-tune all weights for learning. For small laptop experiments, this is slow but simple.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in train_loader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            print(f"epoch={epoch+1} step={global_step} train_loss={loss.item():.4f}")

            if args.eval_every and global_step % args.eval_every == 0:
                eval_loss = evaluate(model, eval_loader)
                ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
                print(f"EVAL step={global_step} eval_loss={eval_loss:.4f} ppl={ppl:.2f}")

            if args.save_every and global_step % args.save_every == 0:
                ckpt_path = out_dir / f"model_step_{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if args.max_steps and global_step >= args.max_steps:
                break

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch={epoch+1} average_train_loss={avg_loss:.4f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    final_path = out_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint: {final_path}")

    print("\nSample generation after fine-tuning:")
    prompt = build_test_prompt()
    text = generate_text_simple(
        model,
        tokenizer,
        prompt,
        max_new_tokens=30,  # FIX 2: Lower generation length for factual QA
        temperature=0.2,  # FIX 2: Lower temperature for more deterministic factual answers
        top_k=40,
        repetition_penalty=1.2,  # FIX 3: Add repetition penalty to reduce repetitive output
        device=DEVICE,
    )
    print(text[len(prompt):].strip())


if __name__ == "__main__":
    main()
