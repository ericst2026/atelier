"""Download models and datasets on an ONLINE machine, in the layout Atelier expects.

    pip install huggingface_hub datasets pyyaml
    python scripts/offline/fetch-materials.py --plan          # what would be fetched, and how big
    python scripts/offline/fetch-materials.py --tracks core   # the usual class set
    python scripts/offline/fetch-materials.py --groups rag eval
    python scripts/offline/fetch-materials.py --names gsm8k Qwen2.5-0.5B

Datasets are written as plain JSONL so the offline node needs no datasets cache
and students can read them with the standard library. Already-complete downloads
are skipped, so re-running after a failure resumes rather than restarts.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Plain HTTP instead of the Xet transfer client. Xet has hung here with a partial
# file and no open connection, cannot resume what it left behind, and fails every
# later download in the process once one errors. HTTP resumes an interrupted file.
# Read when huggingface_hub is imported, so it must be set before the imports below;
# HF_HUB_DISABLE_XET=0 in the environment turns Xet back on.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="./materials")
ap.add_argument("--config", default=str(Path(__file__).with_name("materials.yaml")))
ap.add_argument("--only", choices=["models", "datasets"], help="fetch just one kind")
ap.add_argument("--tracks", nargs="*", help="core, extended, large (default: all)")
ap.add_argument("--groups", nargs="*", help="dataset groups: tokenizer curation pretrain sft preference rl reasoning eval rag tools longcontext safety")
ap.add_argument("--names", nargs="*", help="specific model or dataset names")
ap.add_argument("--plan", action="store_true", help="list what would be fetched and stop")
ap.add_argument("--skip-optional", action="store_true", help="leave out entries marked optional")
args = ap.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
out = Path(args.out)
MODEL_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*", "*.md"]


def wanted(item: dict, kind: str) -> bool:
    if args.names:
        return item["name"] in args.names
    if args.tracks and item.get("track", "core") not in args.tracks:
        return False
    if args.groups and kind == "dataset" and item.get("group") not in args.groups:
        return False
    if args.groups and kind == "model":
        return False
    if args.skip_optional and item.get("optional"):
        return False
    return True


models = [m for m in cfg.get("models", []) if wanted(m, "model")] if args.only != "datasets" else []
datasets = [d for d in cfg.get("datasets", []) if wanted(d, "dataset")] if args.only != "models" else []

total = sum(x.get("size_gb", 0) for x in models + datasets)
print(f"{len(models)} models and {len(datasets)} datasets, about {total:.1f} GB\n")
for m in models:
    print(f"  model    {m['name']:32s} {m.get('size_gb', 0):5.2f} GB  {m['repo']}" + (f"@{m['revision']}" if m.get("revision") else ""))
for d in datasets:
    print(f"  dataset  {d['group']}/{d['name']:24s} {d.get('size_gb', 0):5.2f} GB  {d['repo']}")
if args.plan:
    print("\n--plan: nothing was downloaded.")
    sys.exit(0)

(out / "models").mkdir(parents=True, exist_ok=True)
(out / "datasets").mkdir(parents=True, exist_ok=True)
(out / "hf-cache").mkdir(parents=True, exist_ok=True)
failures = []

if models:
    from huggingface_hub import snapshot_download

    for m in models:
        dest = out / "models" / m["name"]
        # config.json alone proves nothing: the small files land first, so a download
        # that failed on the weights leaves it behind. Only a finished one gets the marker.
        done = dest / ".atelier-complete"
        if done.exists():
            print(f"[skip ] model {m['name']}")
            continue
        print(f"[model] {m['repo']}" + (f"@{m['revision']}" if m.get("revision") else "") + f" → {dest}")
        try:
            snapshot_download(
                repo_id=m["repo"],
                revision=m.get("revision"),
                local_dir=str(dest),
                allow_patterns=m.get("allow_patterns", MODEL_PATTERNS),
                ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.gguf"],
            )
            if not any(dest.glob("*.safetensors")):
                raise RuntimeError("no *.safetensors weights in the snapshot")
            done.touch()
        except Exception as exc:
            print(f"        FAILED: {exc}")
            failures.append(("model", m["name"], str(exc)))

if datasets:
    from datasets import load_dataset

    for d in datasets:
        dest = out / "datasets" / d["group"] / d["name"]
        dest.mkdir(parents=True, exist_ok=True)
        fields = d.get("fields") or {}
        limit = d.get("limit")
        for local_split, remote_split in (d.get("splits") or {"train": "train"}).items():
            path = dest / f"{local_split}.jsonl"
            if path.exists() and path.stat().st_size > 0:
                print(f"[skip ] {d['group']}/{d['name']}/{local_split}")
                continue
            print(f"[data ] {d['repo']} {remote_split} → {path}")
            try:
                # data_dirs: subfolders of the repo read one after another, with `limit`
                # shared equally between them. A stream stops at the limit, so without this
                # a dataset stored folder by folder yields only its first folder.
                parts = d.get("data_dirs") or [None]
                per_part = -(-limit // len(parts)) if limit and d.get("data_dirs") else None
                n = 0
                with open(path, "w", encoding="utf-8") as fh:
                    for part in parts:
                        if d.get("data_files"):
                            # Parquet files named directly; there is one set of files, so every
                            # split in `splits` reads the same ones
                            ds = load_dataset("parquet", data_files={remote_split: d["data_files"]}, split=remote_split, streaming=True)
                        else:
                            ds = load_dataset(d["repo"], d.get("config"), data_dir=part, split=remote_split, streaming=True)
                        k = 0
                        for row in ds:
                            item = {k_: row.get(v) for k_, v in fields.items()} if fields else dict(row)
                            if all(v in (None, "", [], {}) for v in item.values()):
                                continue
                            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                            n += 1
                            k += 1
                            if (per_part and k >= per_part) or (limit and n >= limit):
                                break
                        if part:
                            print(f"        {part}: {k:,} rows")
                        if limit and n >= limit:
                            break
                print(f"        {n:,} rows")
            except Exception as exc:
                print(f"        FAILED: {exc}")
                path.unlink(missing_ok=True)
                if not d.get("optional"):
                    failures.append(("dataset", f"{d['group']}/{d['name']}/{local_split}", str(exc)))

# the generated world's own pools, kept beside the downloaded data
pools_src = Path(__file__).resolve().parents[2] / "experiments" / "tokenizer" / "data" / "pools"
if pools_src.exists():
    pools_dst = out / "datasets" / "tokenizer" / "pools"
    pools_dst.mkdir(parents=True, exist_ok=True)
    for f in pools_src.glob("*.txt"):
        (pools_dst / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[pools] copied to {pools_dst}")

size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
print(f"\nmaterials in {out}: {size / 1e9:.1f} GB")
if failures:
    print(f"\n{len(failures)} required item(s) failed — re-run to retry just those:")
    for kind, name, exc in failures:
        print(f"  {kind} {name}: {exc[:140]}")
    print("\nCommon causes: a dataset needing `huggingface-cli login`, a renamed repo, or a rate limit.")
    sys.exit(1)
