"""Your reader."""
SYSTEM = "Answer the question using only the passages provided. Answer in as few words as possible. If the passages do not contain the answer, reply exactly: no answer."


def answer(question: str, passages: list[dict], generate) -> str:
    body = "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    prompt = f"{body}\n\nQuestion: {question}\nAnswer:" if passages else f"Question: {question}\nAnswer:"
    return generate(prompt, system=SYSTEM, max_new_tokens=64)[0].strip().split("\n")[0]
