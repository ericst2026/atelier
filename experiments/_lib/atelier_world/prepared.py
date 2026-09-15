"""Prepared materials: a teacher's own models and datasets, used instead of generated ones.

Every world experiment can start from what the course generates (the world's text and
tasks, a model from one of your own runs) or from something prepared under
materials/ — a dataset of your own, or a model you trained or downloaded. The step
form lists what is there (param type `material`); this module reads it inside the run.

    from atelier_world.prepared import choose_model, load_lm, read_qa, read_documents

    info = choose_model(P, I, run_key="base_run", run_model_key="model",
                        hint="Choose a Pretraining run (step 3), or a prepared model.")
    lm = load_lm(info, device)                     # Atelier checkpoint or HuggingFace
    outs = lm.generate(prompts, 160, 0.0, system=info["system"])
    rows = read_qa("datasets/our-class/questions", split="train")

Layout under materials/

    models/<name>/model.pt + tokenizer.json [+ meta.json]   an Atelier checkpoint
    models/<name>/config.json + *.safetensors               a HuggingFace model
    datasets/<group>/<name>/<split>.jsonl  (or .txt / .md files for plain text)

meta.json is optional: {"lang": "en", "system": "...", "description": "..."}.

Dataset rows are matched by field name, so most existing JSONL works as it is. The
names must match backend/atelier/materials.py, which builds the list the form shows.
"""
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

TEXT_FIELDS = ("text", "content", "body", "document")
PROMPT_FIELDS = ("prompt", "question", "instruction", "query", "problem", "input")
ANSWER_FIELDS = ("answer", "answers", "answerKey", "output", "response", "target", "completion", "solution")
DATA_SUFFIXES = (".jsonl", ".json")
TEXT_SUFFIXES = (".txt", ".md")


# --- where things are ------------------------------------------------------------
def materials_root() -> Path:
    env = os.environ.get("ATELIER_MATERIALS")
    if env:
        return Path(env)
    try:
        from atelier_sdk import inputs

        md = inputs().get("materials_dir")
        if md:
            return Path(md)
    except Exception:
        pass
    return Path("/srv/atelier/materials")


def material_dir(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").strip().strip("/")
    if not rel:
        raise SystemExit("No prepared material chosen.")
    # checked on the names, not the resolved path: a symlink inside materials/ to a
    # model kept elsewhere is a normal way to share one
    if any(part in ("", ".", "..") for part in rel.split("/")):
        raise SystemExit(f"{rel!r} is not inside the materials folder.")
    p = materials_root() / rel
    if not p.exists():
        raise SystemExit(f"materials/{rel} is not on this machine. On a split install it must be on the worker node that runs the job.")
    return p


def _pick(row: dict, names: Iterable[str]) -> Any:
    for n in names:
        v = row.get(n)
        if v not in (None, "", [], {}):
            return v
    return None


def _as_text(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        # {"text": [...]} (SQuAD answers) or a message
        return _as_text(_pick(v, ("text", "content", "value")) or "")
    if isinstance(v, (list, tuple)):
        return _as_text(v[0]) if v else ""
    return "" if v is None else str(v).strip()


# --- datasets --------------------------------------------------------------------
def dataset_splits(rel: str) -> list[str]:
    d = material_dir(rel)
    return sorted({f.stem for f in d.rglob("*") if f.is_file() and f.suffix.lower() in DATA_SUFFIXES})


def _data_files(rel: str, split: Optional[str]) -> list[Path]:
    d = material_dir(rel)
    files = sorted(f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in DATA_SUFFIXES and not any(p.startswith(".") for p in f.relative_to(d).parts))
    if split:
        chosen = [f for f in files if f.stem == split]
        if not chosen:
            have = sorted({f.stem for f in files})
            raise SystemExit(f"materials/{rel} has no {split!r} split (it has: {', '.join(have) or 'no JSONL files'}).")
        return chosen
    return files


def _rows(files: list[Path], limit: Optional[int]) -> Iterable[tuple[Path, dict]]:
    n = 0
    for f in files:
        with f.open(encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
            fh.seek(0)
            # a .json file may be one array rather than one object per line; JSONL is
            # streamed, since a prepared corpus can be larger than the machine's memory
            if f.suffix.lower() == ".json" and head.lstrip().startswith("["):
                try:
                    source: Iterable[Any] = json.load(fh)
                except ValueError:
                    source = []
            else:
                source = (json.loads(l) for l in fh if l.strip().startswith("{"))
            try:
                for row in source:
                    if isinstance(row, dict):
                        yield f, row
                        n += 1
                        if limit and n >= limit:
                            return
            except ValueError as exc:
                raise SystemExit(f"{f.name}: not valid JSON Lines ({exc}).")


def read_documents(rel: str, limit: Optional[int] = None, split: Optional[str] = None) -> list[dict[str, Any]]:
    """Plain text: one document per .txt/.md file, or per JSONL row with a text field."""
    d = material_dir(rel)
    docs: list[dict[str, Any]] = []
    for f, row in _rows(_data_files(rel, split), limit):
        text = _as_text(_pick(row, TEXT_FIELDS))
        if text:
            docs.append({"id": len(docs), "text": text, "kind": row.get("kind") or f.stem, "truth": "unknown", "file": f.name})
    if not split:
        for f in sorted(d.rglob("*")):
            if limit and len(docs) >= limit:
                break
            if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES and f.name.lower() != "readme.md":
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    docs.append({"id": len(docs), "text": text, "kind": f.suffix.lstrip("."), "truth": "unknown", "file": f.name})
    if not docs:
        raise SystemExit(f"materials/{rel} has no text in it: expected .txt/.md files, or JSONL rows with one of {', '.join(TEXT_FIELDS)}.")
    return docs[:limit] if limit else docs


def _reply(v: Any) -> str:
    """An answer that may be a chat: [{"role": "user", ...}, {"role": "assistant", ...}]
    means the last assistant turn, not the first message."""
    if isinstance(v, list) and v and all(isinstance(m, dict) and "role" in m for m in v):
        said = [m for m in v if m.get("role") == "assistant"] or v[-1:]
        return _as_text(said[-1])
    return _as_text(v)


def qa_row(row: dict, i: int = 0) -> Optional[dict[str, Any]]:
    """{prompt, answer, steps, family, difficulty} from a row with any of the usual names.

    Also reads the two common shapes that would otherwise grade wrong: a worked
    solution ending in "#### 72" (GSM8K) gives the answer 72 and the lines above it as
    steps, and multiple choice ({"choices": [...], "answer": 1}) lists the options as
    A–D in the prompt and grades the letter."""
    prompt = _as_text(_pick(row, ("prompt", "question", "instruction", "query", "problem")))
    extra = _as_text(row.get("input") or row.get("context")) if row.get("instruction") else ""
    if not prompt:
        prompt = _as_text(row.get("input"))
        extra = ""
    if extra:
        prompt = f"{prompt}\n{extra}"
    raw_answer = _pick(row, ANSWER_FIELDS)
    answer = _reply(raw_answer)
    steps = row.get("steps") or row.get("rationale") or []
    choices, labels = row.get("choices"), []
    if isinstance(choices, dict):
        # {"text": [...], "label": ["A", "B", ...] or ["1", "2", ...]} (ARC)
        labels, choices = [str(l) for l in choices.get("label") or []], choices.get("text")
    if isinstance(choices, list) and 1 < len(choices) <= 10 and all(isinstance(c, str) for c in choices):
        letters = "ABCDEFGHIJ"[: len(choices)]
        prompt = prompt + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        if isinstance(raw_answer, int) and 0 <= raw_answer < len(choices):
            answer = letters[raw_answer]
        elif answer in labels and len(labels) == len(choices):
            answer = letters[labels.index(answer)]
        elif answer in choices:
            answer = letters[choices.index(answer)]
    elif "\n#### " in f"\n{answer}":
        work, _, final = f"\n{answer}".rpartition("\n#### ")
        answer = final.strip()
        steps = steps or [re.sub(r"<<[^>]*>>", "", s).strip() for s in work.splitlines() if s.strip()]
    if not prompt or not answer:
        return None
    if isinstance(steps, str):
        steps = [s for s in steps.splitlines() if s.strip()]
    return {
        "id": row.get("id", i),
        "prompt": prompt,
        "answer": answer,
        "steps": list(steps),
        "family": str(row.get("family") or row.get("category") or row.get("subject") or "prepared"),
        "difficulty": row.get("difficulty", 1) if isinstance(row.get("difficulty"), int) else 1,
    }


def read_qa(rel: str, split: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Questions with gradable answers, e.g. {"question": ..., "answer": ...}."""
    out = [q for i, (_, row) in enumerate(_rows(_data_files(rel, split), limit)) if (q := qa_row(row, i))]
    if not out:
        raise SystemExit(f"materials/{rel}{f' ({split})' if split else ''} has no usable rows: each needs a question ({', '.join(PROMPT_FIELDS)}) and an answer ({', '.join(ANSWER_FIELDS)}).")
    return out


def read_pairs(rel: str, split: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Preference pairs: {prompt, chosen, rejected}."""
    out = []
    for _, row in _rows(_data_files(rel, split), limit):
        prompt = _as_text(_pick(row, PROMPT_FIELDS))
        chosen, rejected = _reply(row.get("chosen")), _reply(row.get("rejected"))
        if prompt and chosen and rejected:
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "answer": _as_text(_pick(row, ANSWER_FIELDS)) or None})
    if not out:
        raise SystemExit(f"materials/{rel} has no preference pairs: each row needs a prompt, a chosen and a rejected answer.")
    return out


def train_val(rel: str, reader: Callable[..., list[dict]], val_size: int, seed: int = 0, limit: Optional[int] = None) -> tuple[list[dict], list[dict]]:
    """The dataset's own train and validation/test splits when it has them; otherwise
    a held-out slice of what there is, so evaluation never grades training rows."""
    splits = set(dataset_splits(rel))
    val_name = next((s for s in ("validation", "val", "dev", "test") if s in splits), None)
    if "train" in splits and val_name:
        return reader(rel, split="train", limit=limit), reader(rel, split=val_name, limit=val_size)
    rows = reader(rel, limit=limit)
    random.Random(seed).shuffle(rows)
    k = min(val_size, max(1, len(rows) // 5))
    if len(rows) < 2:
        raise SystemExit(f"materials/{rel} needs at least two rows to hold one out for evaluation.")
    return rows[k:], rows[:k]


# --- models ----------------------------------------------------------------------
def model_format(d: Path) -> str:
    if (d / "model.pt").exists():
        return "atelier"
    if (d / "config.json").exists():
        return "hf"
    raise SystemExit(f"{d.name} is neither an Atelier checkpoint (model.pt + tokenizer.json) nor a HuggingFace model (config.json + weights).")


def choose_model(P: dict, I: dict, *, run_key: str, run_model_key: str, hint: str, source_key: str = "model_source", material_key: str = "model_material", formats: tuple[str, ...] = ("atelier", "hf")) -> dict[str, Any]:
    """The model a step starts from: a run of yours (generated), or one prepared under
    materials/models. Returns {format, model, tokenizer, lang, system, label, outputs}.

    `outputs` holds the chosen run's outputs (empty for a prepared model), so a step
    can still pass along anything else the run produced."""
    source = str(P.get(source_key) or "generated")
    if source != "prepared":
        ref = I.get(f"{run_key}_run")
        if not ref:
            raise SystemExit(hint)
        o = ref["outputs"]
        info = {
            "source": "generated",
            "format": o.get("model_format", "atelier"),
            "model": o[run_model_key],
            "tokenizer": o.get("tokenizer"),
            "lang": o.get("lang", "en"),
            "system": o.get("system"),
            "adapter": o.get("adapter"),
            "label": f"run #{ref['id']}",
            "outputs": o,
        }
    else:
        rel = P.get(material_key)
        if not rel:
            raise SystemExit("Choose a prepared model from the list, or switch the source back to one of your runs.")
        d = material_dir(rel)
        fmt = model_format(d)
        meta = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if fmt == "atelier" and not (d / "tokenizer.json").exists():
            raise SystemExit(f"materials/{rel} has model.pt but no tokenizer.json. Copy the tokenizer the model was trained with next to it.")
        info = {
            "source": "prepared",
            "format": fmt,
            "model": str(d / "model.pt") if fmt == "atelier" else str(d),
            "tokenizer": str(d / "tokenizer.json") if fmt == "atelier" else str(d),
            "lang": meta.get("lang") or P.get("lang") or "en",
            "system": meta.get("system"),
            "adapter": None,
            "label": f"materials/{rel}",
            "outputs": {},
        }
    if info["format"] not in formats:
        kinds = {"atelier": "an Atelier checkpoint", "hf": "a HuggingFace model"}
        raise SystemExit(f"This step works on {' or '.join(kinds[f] for f in formats)}; {info['label']} is {kinds.get(info['format'], info['format'])}. The experiment looks inside the model's own layers, which only the Atelier model exposes the way it needs.")
    return info


def model_outputs(info: dict, key: str) -> dict[str, Any]:
    """What a step passes on so the next step (or experiment) finds the same model."""
    return {key: info["model"], "tokenizer": info["tokenizer"], "lang": info["lang"], "system": info.get("system"), "model_format": info["format"], "adapter": info.get("adapter"), "model_label": info["label"]}


class LM:
    """One interface over both kinds of model, for the parts every experiment shares:
    generating answers. `model`/`tok` are the underlying objects for anything more."""

    def __init__(self, info: dict, device: str, dtype: str = "bf16"):
        self.info, self.format, self.device = info, info["format"], device
        self.ck: dict[str, Any] = {}
        if self.format == "atelier":
            from atelier_mini.model import MiniLM
            from atelier_mini.tok import load_tokenizer

            self.model, self.ck = MiniLM.load(info["model"], device)
            self.tok = load_tokenizer(info["tokenizer"])
        else:
            from atelier_nlp import hf

            self.tok = hf.load_tokenizer(info["model"])
            # a LoRA adapter from a HuggingFace fine-tuning run is merged in at load time
            self.model = hf.load_model(info["model"], dtype=dtype if device == "cuda" else "fp32", device=device, adapter=info.get("adapter"))
        self.system = info.get("system") or self.ck.get("system")

    def generate(self, prompts: list[str], max_new_tokens: int = 160, temperature: float = 0.0, num_samples: int = 1, batch_size: int = 32, system: Optional[str] = None, top_k: int = 0, progress: Optional[Callable[[int, int], None]] = None) -> list[list[str]]:
        system = system if system is not None else self.system
        if self.format == "atelier":
            from atelier_mini.gen import generate

            return generate(self.model, self.tok, prompts, max_new_tokens, temperature, top_k=top_k, num_samples=num_samples, batch_size=batch_size, system=system, progress=progress)
        from atelier_nlp import hf

        chat = [hf.chat_prompt(self.tok, p, system) for p in prompts]
        return hf.generate_batch(self.model, self.tok, chat, max_new_tokens, temperature, batch_size=max(1, batch_size // max(1, num_samples)), num_return_sequences=num_samples, progress=progress)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def load_lm(info: dict, device: str) -> LM:
    return LM(info, device)
