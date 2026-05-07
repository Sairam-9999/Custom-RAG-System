import torch
import tiktoken
from pathlib import Path

from custom_gpt_updated import GPTModel, GPT_CONFIG_124M


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "rag_finetuned_gpt2_ckpt" / "model_final.pt"

MAX_NEW_TOKENS = 30
TEMPERATURE = 0.2


def load_model():
    model = GPTModel(GPT_CONFIG_124M).to(DEVICE)

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def build_prompt(context, question):
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


@torch.no_grad()
def generate(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    eos_id = tokenizer.encode(
        "<|endoftext|>",
        allowed_special={"<|endoftext|>"}
    )[0]

    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_tensor)
        next_token_logits = logits[:, -1, :] / TEMPERATURE

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        if next_token.item() == eos_id:
            break

        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    output_ids = input_tensor[0].tolist()
    answer_ids = output_ids[len(input_ids):]
    answer = tokenizer.decode(answer_ids)

    return answer.strip()


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model()

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

    questions = [
        "What is the date range for The Red-Headed League?",
        "What insight does The Red-Headed League show?",
        "What is Watson's age?",
    ]

    for q in questions:
        prompt = build_prompt(context, q)
        answer = generate(model, tokenizer, prompt)

        print("\nQUESTION:", q)
        print("ANSWER:", answer)


if __name__ == "__main__":
    main()
