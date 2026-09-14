"""Train an embedder, then measure it.  python project/train.py --model <model.pt> --tokenizer <tok.json>"""
import argparse
from pathlib import Path

import torch

import retriever
from atelier_mini.embed import BM25, Index, MiniEmbedder, recall_at_k, train_contrastive
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import Library, World

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--docs", type=int, default=8000)
ap.add_argument("--pairs", type=int, default=20000)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--temperature", type=float, default=0.05)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=4242)
lib = Library(world, n_docs=args.docs, seed=4242)
texts = [d["text"] for d in lib.documents]
queries = lib.queries(600)
pairs = lib.pairs(args.pairs)

bm = BM25(texts)
for p in pairs:
    wrong = [i for i, _ in bm.search(p["query"], 3) if i != p["gold_id"]]
    if wrong:
        p["hard_negative"] = texts[wrong[0]]

tok = MiniTokenizer.load(args.tokenizer)
model, _ = MiniLM.load(args.model, device)
emb = MiniEmbedder(model, tok).to(device)


def recall5():
    res = Index(emb.encode(texts, batch_size=256), [{"doc_id": i} for i in range(len(texts))]).search(emb.encode([q["query"] for q in queries], batch_size=256), 5)
    return recall_at_k(res, "doc_id", [q["gold_id"] for q in queries], 5)


print(f"BM25 recall@5:      {sum(1 for q in queries if q['gold_id'] in [i for i, _ in bm.search(q['query'], 5)]) / len(queries):.1%}")
print(f"zero-shot recall@5: {recall5():.1%}")
out = Path("outputs")
train_contrastive(emb, pairs[500:], out, pairs[:500], epochs=args.epochs, batch_size=args.batch, temperature=args.temperature, hard_negatives=True, device=device,
                  on_log=lambda r: print(f"step {r['step']} " + (f"in-batch {r['in_batch_accuracy']:.0%}" if "in_batch_accuracy" in r else f"loss {r.get('loss', 0):.3f}"), flush=True))
torch.save({"model_state": emb.model.state_dict(), "config": emb.model.config.to_dict(), "proj": None, "layer": emb.layer, "dim": emb.dim, "max_length": emb.max_length, "tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang}, out / "embedder.pt")
print(f"trained recall@5:   {recall5():.1%}  → outputs/embedder.pt")
