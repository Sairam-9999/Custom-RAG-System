from pathlib import Path
import sys
import torch
import tiktoken


# Project root = parent folder of rag/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

# Allow importing custom_gpt_updated.py from the model/ folder
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from custom_gpt_updated import GPTModel, GPT_CONFIG_124M


DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "model" / "rag_finetuned_gpt2_ckpt" / "model_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = tiktoken.get_encoding("gpt2")
EOS_ID = TOKENIZER.encode(
    "<|endoftext|>",
    allowed_special={"<|endoftext|>"},
)[0]


def build_finetuned_prompt(context: str, question: str) -> str:
    """
    This prompt format must match the format used during fine-tuning.
    """
    return f"""### Instruction:
Answer the question using only the provided context.
Do not use outside knowledge.
If the answer is not in the context, say: I don't know from the provided context.

### Context:
{context}

### Question:
{question}

### Answer:
"""


def load_finetuned_model(checkpoint_path: str | None = None):
    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH

    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {ckpt_path}\n"
            "Expected default path: model/rag_finetuned_gpt2_ckpt/model_final.pt"
        )

    model = GPTModel(GPT_CONFIG_124M).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def generate_from_model(
    model,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    temperature = max(temperature, 1e-5)

    input_ids = TOKENIZER.encode(prompt)
    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    generated_text = ""

    for _ in range(max_new_tokens):
        idx_cond = input_tensor[:, -GPT_CONFIG_124M["context_length"]:]
        logits = model(idx_cond)

        next_token_logits = logits[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            unique_tokens, counts = torch.unique(input_tensor, return_counts=True)
            penalty = torch.ones_like(next_token_logits)
            for token, count in zip(unique_tokens, counts):
                penalty[0, token] = repetition_penalty ** count
            next_token_logits = next_token_logits / penalty

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_vals, _ = torch.topk(next_token_logits, top_k)
            min_val = top_vals[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(
                next_token_logits < min_val,
                torch.tensor(-float("inf"), device=DEVICE),
                next_token_logits
            )

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

        # Sample from the filtered distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == EOS_ID:
            break

        input_tensor = torch.cat([input_tensor, next_token], dim=1)

        # Decode current generated text for anti-repetition check
        output_ids = input_tensor[0].tolist()
        answer_ids = output_ids[len(input_ids):]
        generated_text = TOKENIZER.decode(answer_ids).replace("", "").strip()

        # Anti-repetition stopping: stop if same phrase repeats more than 2 times
        sentences = generated_text.split(". ")
        if len(sentences) > 1:
            last_phrase = sentences[-1].strip()
            if last_phrase and generated_text.lower().count(last_phrase.lower()) > 2:
                break

    output_ids = input_tensor[0].tolist()
    answer_ids = output_ids[len(input_ids):]
    answer = TOKENIZER.decode(answer_ids)

    return answer.replace("", "").strip()


def generate_answer(
    context: str,
    question: str,
    checkpoint_path: str | None = None,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Main function used by main.py.

    Inputs:
        context: retrieved RAG chunks joined as text
        question: user question

    Output:
        grounded answer generated by your fine-tuned GPT-2 model
    """
    prompt = build_finetuned_prompt(context=context, question=question)
    model = load_finetuned_model(checkpoint_path=checkpoint_path)

    return generate_from_model(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
