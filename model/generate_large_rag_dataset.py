import json
import random
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

EOS = "<|endoftext|>"
REFUSAL = "I don't know from the provided context."
TARGET_TOTAL = 3000
EVAL_RATIO = 0.12

random.seed(42)


FACT_INST = (
    "Answer using only the context. Return the shortest correct answer. "
    "Do not copy the full context sentence. If the answer is not in the context, "
    "say: I don't know from the provided context."
)

REASON_INST = (
    "Answer using only the context. Explain clearly in 1-2 short sentences. "
    "Do not invent facts. If the answer is not in the context, "
    "say: I don't know from the provided context."
)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make(context, question, answer, typ="fact"):
    inst = REASON_INST if typ == "reasoning" else FACT_INST
    answer = answer.strip()
    if not answer.endswith(EOS):
        answer += EOS

    return {
        "type": typ,
        "prompt": (
            f"### Instruction:\n{inst}\n\n"
            f"### Context:\n{context.strip()}\n\n"
            f"### Question:\n{question.strip()}\n\n"
            f"### Answer:\n"
        ),
        "completion": answer,
    }


def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip().split()) >= 5
    ]


def chunk_sentences(sentences, chunk_size=650, overlap=2):
    chunks = []
    current = []

    for sent in sentences:
        candidate = " ".join(current + [sent])
        if len(candidate) <= chunk_size:
            current.append(sent)
        else:
            if current:
                chunks.append(" ".join(current))
            current = current[-overlap:] + [sent]

    if current:
        chunks.append(" ".join(current))

    return chunks


def clean_answer(ans):
    ans = ans.strip(" .,:;!?\"'")
    return ans[:1].upper() + ans[1:] + "." if ans else ""


def extract_entities(sentence):
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)


def extract_after(pattern, sentence):
    m = re.search(pattern, sentence, re.IGNORECASE)
    if not m:
        return None
    return clean_answer(m.group(1))


def question_variants(q):
    base = q.strip().rstrip("?")
    variants = {base + "?"}

    starters = {
        "what": ["What exactly", "Which thing"],
        "who": ["Which person", "Who exactly"],
        "where": ["In what place", "What location"],
        "when": ["At what time", "When exactly"],
        "why": ["For what reason", "Why exactly"],
        "how": ["In what way", "How exactly"],
        "which": ["Which one", "What"],
        "did": ["Did the context say", "Was it true that"],
        "was": ["Was it true that", "Did the context say"],
    }

    lower = base.lower()
    for starter, replacements in starters.items():
        if lower.startswith(starter):
            rest = base[len(starter):].strip()
            for rep in replacements:
                variants.add(f"{rep} {rest}?")
            break

    variants.add("Using the context, " + base[:1].lower() + base[1:] + "?")
    return list(variants)


def noisy_context(answer_ctx, all_contexts):
    distractors = random.sample(all_contexts, k=min(2, len(all_contexts)))
    chunks = [
        f"--- Chunk 1 ---\n{answer_ctx}",
    ]
    for i, d in enumerate(distractors, start=2):
        chunks.append(f"--- Chunk {i} ---\n{d}")

    random.shuffle(chunks)
    return "\n\n".join(chunks)


def generate_from_sentence(sentence, context):
    examples = []
    entities = extract_entities(sentence)

    # WHO examples
    if entities:
        first = entities[0]
        examples.append(make(context, f"Who is mentioned in this passage?", clean_answer(first), "fact"))

    # WHERE examples
    where_patterns = [
        r"\b(?:inside|in|at|above|beneath|beyond|across|through|under)\s+([^.!?]+)",
    ]
    for pat in where_patterns:
        ans = extract_after(pat, sentence)
        if ans and len(ans.split()) <= 18:
            examples.append(make(context, "Where does this happen?", ans, "fact"))

    # WHEN examples
    when_patterns = [
        r"\b(every night[^.!?]*)",
        r"\b(some nights[^.!?]*)",
        r"\b(after sunset[^.!?]*)",
        r"\b(before [^.!?]+)",
        r"\b(after [^.!?]+)",
    ]
    for pat in when_patterns:
        ans = extract_after(pat, sentence)
        if ans:
            examples.append(make(context, "When does this happen?", ans, "fact"))

    # WHAT examples
    what_patterns = [
        r"(.+?)\s+(?:destroyed|carried|contained|revealed|illuminated|swallowed|crossed|remembered|waited|slept|chanted)\b",
        r"\b(?:contained|revealed|illuminated|swallowed|carried)\s+([^.!?]+)",
        r"\bwas\s+([^.!?]+)",
        r"\bwere\s+([^.!?]+)",
    ]
    for pat in what_patterns:
        ans = extract_after(pat, sentence)
        if ans and 1 <= len(ans.split()) <= 20:
            examples.append(make(context, "What is stated in this passage?", ans, "fact"))

    # WHICH examples
    if len(entities) >= 2:
        examples.append(make(context, "Which named group or person is important here?", clean_answer(entities[-1]), "fact"))

    # DID / WAS boolean-style examples
    if entities:
        examples.append(make(context, f"Did the passage mention {entities[0]}?", "Yes.", "fact"))
    examples.append(make(context, "Was this answer supported by the context?", "Yes.", "fact"))

    # WHY examples
    if "because" in sentence.lower():
        ans = extract_after(r"because\s+([^.!?]+)", sentence)
        if ans:
            examples.append(make(context, "Why did this happen?", ans, "reasoning"))

    if "that was why" in sentence.lower():
        examples.append(make(context, "Why did the character act this way?", sentence.strip(), "reasoning"))

    # HOW examples
    if any(w in sentence.lower() for w in ["like", "faster than", "through", "slowly"]):
        examples.append(make(context, "How is this described?", sentence.strip(), "reasoning"))

    return examples


def generate_refusals(context):
    refusal_questions = [
        "What is the exact age of the main character?",
        "Who is the main character's mother?",
        "Where was the main character born?",
        "When exactly was the final battle fought?",
        "Why did the villain choose that exact date?",
        "How many soldiers were there exactly?",
        "Which city was the capital?",
        "Did the hero have a secret brother?",
        "Was the treasure made of gold?",
        "What was the exact year?",
        "Who guarded the treasure chamber?",
        "Where was the hidden treasure buried?",
        "When was the prophecy first written?",
        "How much money did the hero receive?",
        "Which weapon had a name?",
    ]
    return [make(context, q, REFUSAL, "refusal") for q in refusal_questions]


def main():
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}")

    examples = []
    all_contexts = []

    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences)

        all_contexts.extend(chunks)

        for chunk in chunks:
            chunk_sents = split_sentences(chunk)

            for sent in chunk_sents:
                examples.extend(generate_from_sentence(sent, chunk))

            # generic chunk-level questions
            entities = extract_entities(chunk)
            if entities:
                examples.append(make(chunk, "Who is a key named person or group in this context?", clean_answer(entities[0]), "fact"))

            examples.append(make(chunk, "What is this passage mainly about?", chunk_sents[0] if chunk_sents else chunk, "reasoning"))

            if random.random() < 0.35:
                examples.extend(generate_refusals(chunk))

    # Augment with question variants and noisy RAG-style contexts
    augmented = []
    for ex in examples:
        prompt = ex["prompt"]
        ctx_match = re.search(r"### Context:\n(.*?)\n\n### Question:", prompt, re.S)
        q_match = re.search(r"### Question:\n(.*?)\n\n### Answer:", prompt, re.S)

        if not ctx_match or not q_match:
            continue

        ctx = ctx_match.group(1).strip()
        q = q_match.group(1).strip()
        ans = ex["completion"].replace(EOS, "").strip()
        typ = ex.get("type", "fact")

        for qv in question_variants(q):
            augmented.append(make(ctx, qv, ans, typ))

            if len(all_contexts) >= 3:
                augmented.append(make(noisy_context(ctx, all_contexts), qv, ans, typ))

            compact = "Relevant passage:\n" + ctx + "\nUse only this passage."
            augmented.append(make(compact, qv, ans, typ))

    examples.extend(augmented)

    # Deduplicate
    seen = set()
    unique = []
    for ex in examples:
        key = (ex["prompt"], ex["completion"])
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    random.shuffle(unique)
    unique = unique[:TARGET_TOTAL]

    eval_size = max(100, int(len(unique) * EVAL_RATIO))
    eval_rows = unique[:eval_size]
    train_rows = unique[eval_size:]

    write_jsonl(MODEL_DIR / "rag_train_mixed_large.jsonl", train_rows)
    write_jsonl(MODEL_DIR / "rag_eval_mixed_large.jsonl", eval_rows)
    write_jsonl(MODEL_DIR / "rag_training_mixed_large_all.jsonl", train_rows + eval_rows)

    manifest = {
        "total_examples": len(unique),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "type_counts": dict(Counter(ex.get("type", "unknown") for ex in unique)),
        "sources": [p.name for p in txt_files],
        "question_types_supported": ["what", "who", "where", "when", "why", "how", "which", "did", "was"],
        "notes": "Fully generic synthetic RAG dataset generator using sentence extraction, noisy contexts, question variants, and refusals.",
    }

    (MODEL_DIR / "rag_mixed_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()