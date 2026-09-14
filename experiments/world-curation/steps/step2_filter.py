"""Step 2 — thresholds, scored against the labels."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl, write_jsonl
from atelier_world import score_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_signals import RULES, apply_rules  # noqa: E402

parse_args()
P = params({"min_chars": 120, "max_symbol_ratio": 0.2, "max_digit_ratio": 0.25, "max_upper_ratio": 0.4, "max_dup_line_ratio": 0.4, "min_stopword_hits": 2, "max_ellipsis_ratio": 0.3, "min_unique_word_ratio": 0.35})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
docs = read_jsonl(I["corpus"])
progress(10, f"{len(docs):,} documents")

decisions = [apply_rules(d["text"], P) for d in docs]
kept_flags = [k for k, _ in decisions]
why = Counter(r for k, r in decisions if not k)
sc = score_filter(docs, kept_flags)
kept = [d for d, k in zip(docs, kept_flags) if k]
write_jsonl(run_dir / "filtered.jsonl", kept)
(run_dir / "filter_params.json").write_text(json.dumps(dict(P), indent=2))
progress(55, "measuring each rule on its own")

solo = []
probe = docs[:6000]
base_keep = sum(1 for d in probe if apply_rules(d["text"], P)[0])
for rule in RULES:
    relaxed = dict(P)
    relaxed.update({
        "too short": {"min_chars": 0}, "symbols": {"max_symbol_ratio": 1.0}, "digits": {"max_digit_ratio": 1.0},
        "shouting": {"max_upper_ratio": 1.0}, "repeated lines": {"max_dup_line_ratio": 1.0},
        "not prose": {"min_stopword_hits": 0}, "truncated": {"max_ellipsis_ratio": 1.0},
        "few distinct words": {"min_unique_word_ratio": 0.0},
    }[rule])
    without = sum(1 for d in probe if apply_rules(d["text"], relaxed)[0])
    caught = [d for d in probe if apply_rules(d["text"], relaxed)[0] and not apply_rules(d["text"], P)[0]]
    solo.append({"rule": rule, "removed": why.get(rule, 0) / len(docs), "uniquely": (without - base_keep) / len(probe), "of_which_good": sum(1 for d in caught if d["clean"]) / max(len(caught), 1)})

by_kind = sc["by_kind"]
lost = [d for d, k in zip(docs, kept_flags) if not k and d["clean"]]
survived = [d for d, k in zip(docs, kept_flags) if k and not d["clean"]]

R = Result()
R.metric("f1", "F1 against the labels", sc["f1"], "num", "kept")
R.metric("precision", "Precision", sc["precision"], "pct", "sky", help="of what you kept, how much was actually clean")
R.metric("recall", "Recall", sc["recall"], "pct", "hold", help=f"of the clean documents, how many survived — you lost {len(lost):,}")
R.metric("kept", "Documents kept", sc["kept"], "int", "raw", help=f"{sc['kept'] / len(docs):.0%} of the corpus")
R.chart("by_kind", "What survived, by kind", [{"kind": k, "kept": v, "target": 1.0 if k == "clean" else 0.0} for k, v in by_kind.items()], "kind", [{"key": "kept", "label": "Kept", "color": "raw"}, {"key": "target", "label": "Should keep", "color": "kept"}], "bar", y_domain=[0, 1], note="The clean bar should be tall and every other bar short. Duplicates will stay tall whatever you do here — they are good text, and the next step is where they go.")
R.chart("rules", "What each rule contributes", solo, "rule", [{"key": "removed", "label": "Removed by this rule first", "color": "dup"}, {"key": "uniquely", "label": "Only this rule catches it", "color": "raw"}, {"key": "of_which_good", "label": "Of those, clean", "color": "hold"}], "bar", note="A rule with a tall third bar is doing damage: the documents only it removes were mostly fine.")
R.table("lost", "Clean documents you removed", [{"key": "why", "label": "Rule"}, {"key": "text", "label": "Text"}], [{"why": apply_rules(d["text"], P)[1], "text": d["text"][:260]} for d in lost[:20]], note="If these look fine to you, a threshold is too tight.")
R.table("survived", "Spoiled documents that survived", [{"key": "kind", "label": "Kind"}, {"key": "text", "label": "Text"}], [{"kind": d["kind"], "text": d["text"][:260]} for d in survived[:15]])
R.artifact(run_dir / "filtered.jsonl", "filtered.jsonl")
R.output("filtered", str(run_dir / "filtered.jsonl")).output("corpus", I["corpus"]).output("f1", sc["f1"]).output("filter_params", str(run_dir / "filter_params.json"))
for k in ("lang", "seed"):
    if k in I:
        R.output(k, I[k])
R.save()
