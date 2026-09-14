"""Run your system end to end.  python project/run.py --n 200"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "rag"))
from answer import answer  # noqa: E402
from atelier_nlp import hf  # noqa: E402
from lib_qa import exact_match, is_refusal  # noqa: E402
from retriever import build, search  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--articles", type=int, default=10000)
ap.add_argument("--reader", default="Qwen2.5-0.5B-Instruct")
args = ap.parse_args()

corpus = hf.dataset_split("rag", "wikipedia-simple", "train", limit=args.articles)
index = build(corpus)
mp = hf.model_path(args.reader)
tok, model = hf.load_tokenizer(mp), hf.load_model(mp)


def generate(prompt, system=None, max_new_tokens=64):
    return hf.generate_batch(model, tok, [hf.chat_prompt(tok, prompt, system)], max_new_tokens, 0.0)[0]


squad = hf.dataset_split("rag", "squad-v2", "validation", limit=4000)
qs = [q for q in squad if (q.get("answers") or {}).get("text")][: args.n]
correct = 0
for i, q in enumerate(qs):
    passages = search(index, q["question"], 5)
    pred = answer(q["question"], passages[:3], generate)
    ok = exact_match(pred, q["answers"]["text"])
    correct += ok
    if i < 10:
        print(f"{'✓' if ok else '✗'} {q['question'][:70]!r} → {pred!r}")
print(f"\nexact match {correct / len(qs):.1%} on {len(qs)} questions")
