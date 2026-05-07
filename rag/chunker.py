import re


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text, chunk_size=650, overlap=3):
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []

    for para in paragraphs:
        sentences = split_sentences(para)

        current = []

        for sentence in sentences:
            temp = " ".join(current + [sentence])

            if len(temp) <= chunk_size:
                current.append(sentence)
            else:
                if current:
                    chunks.append(" ".join(current))

                current = current[-overlap:] + [sentence]

        if current:
            chunks.append(" ".join(current))

    return chunks
