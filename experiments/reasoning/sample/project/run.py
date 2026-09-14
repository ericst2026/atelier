"""Run the solver on training problems with a metered generate().  python project/run.py --n 50"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier_nlp import hf  # noqa: E402
from solve import solve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
args = ap.parse_args()
mp = hf.model_path(args.model)
tok, model = hf.load_tokenizer(mp), hf.load_model(mp)
meter = {"tokens": 0, "calls": 0}


def generate(prompt, n=1, temperature=0.0, max_new_tokens=384):
    outs = hf.generate_batch(model, tok, [hf.chat_prompt(tok, prompt)], max_new_tokens, temperature, num_return_sequences=n)[0]
    meter["tokens"] += sum(len(tok(o).input_ids) for o in outs)
    meter["calls"] += 1
    return outs


rows = hf.dataset_split("rl", "gsm8k", "train")
random.Random(3).shuffle(rows)
correct = 0
for i, r in enumerate(rows[: args.n]):
    pred = solve(r["question"], generate)
    ok = hf.answers_equal(pred, hf.gsm8k_gold(r["answer"]))
    correct += ok
    print(f"{i + 1:3d} {'✓' if ok else '✗'} pred={pred!r} gold={hf.gsm8k_gold(r['answer'])!r}")
print(f"accuracy {correct / args.n:.2%} · {meter['tokens'] / args.n:.0f} tokens/problem · {meter['calls'] / args.n:.1f} calls/problem")
