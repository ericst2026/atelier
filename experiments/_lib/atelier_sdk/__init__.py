"""Tiny SDK for experiment steps, graders and student code.

    from atelier_sdk import params, inputs, progress, Result, material

A step reads its parameters from <run_dir>/params.json, the outputs of the
previous step from <run_dir>/inputs.json, reports progress by printing
`::progress {...}` lines, and writes <run_dir>/result.json through Result.
Nothing here imports torch, so it is safe everywhere."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = ["RUN_DIR", "MATERIALS", "params", "inputs", "progress", "Result", "material", "hist", "read_jsonl", "write_jsonl", "parse_args", "run_dir"]


def _run_dir() -> Path:
    return Path(os.environ.get("ATELIER_RUN_DIR") or ".").resolve()


RUN_DIR = _run_dir()
MATERIALS = Path(os.environ.get("ATELIER_MATERIALS") or "materials").resolve()


def parse_args(extra: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    """Accept --run-dir (and --project for graders); sets RUN_DIR."""
    global RUN_DIR
    ap = extra or argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--project", default=None)
    ns, _ = ap.parse_known_args()
    if ns.run_dir:
        os.environ["ATELIER_RUN_DIR"] = ns.run_dir
        RUN_DIR = Path(ns.run_dir).resolve()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
    return ns


def run_dir() -> Path:
    return _run_dir()


def _load(name: str) -> dict[str, Any]:
    p = _run_dir() / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def params(defaults: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out = dict(defaults or {})
    out.update({k: v for k, v in _load("params.json").items() if v is not None})
    return out


def inputs() -> dict[str, Any]:
    return _load("inputs.json")


def material(rel: str, must_exist: bool = True) -> Path:
    p = MATERIALS / rel
    if must_exist and not p.exists():
        raise FileNotFoundError(f"material not found: {p} (copy it to the materials folder on the server)")
    return p


_last_progress = 0.0


def progress(pct: Optional[float] = None, msg: Optional[str] = None, min_interval: float = 0.0, **series: Any) -> None:
    """Report progress. Extra keywords become live chart series (use step=... for the x value)."""
    global _last_progress
    now = time.time()
    if min_interval and now - _last_progress < min_interval and not series:
        return
    _last_progress = now
    payload: dict[str, Any] = {}
    if pct is not None:
        payload["pct"] = round(float(pct), 2)
    if msg is not None:
        payload["msg"] = str(msg)
    if series:
        payload["series"] = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in series.items()}
    print("::progress " + json.dumps(payload), flush=True)


def hist(values: Iterable[float], bins: int = 20, log: bool = False, lo: Optional[float] = None, hi: Optional[float] = None, integer: bool = False) -> list[dict[str, Any]]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if log:
        lo = max(lo, 1e-9)
        edges = [lo * (hi / lo) ** (i / bins) for i in range(bins + 1)] if hi > lo else [lo, hi]
    else:
        if integer and hi - lo < bins:
            bins = max(1, int(hi - lo) + 1)
            edges = [lo + i for i in range(bins + 1)]
        else:
            edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in vals:
        if log:
            i = int(math.floor(math.log(v / lo) / math.log(hi / lo) * bins)) if hi > lo else 0
        else:
            i = int((v - lo) / (hi - lo) * bins) if hi > lo else 0
        counts[min(max(i, 0), bins - 1)] += 1

    def label(a: float, b: float) -> str:
        if integer:
            return f"{int(a)}" if bins == int(hi - lo) + 1 else f"{int(a)}–{int(b)}"
        return f"{a:.3g}–{b:.3g}"

    return [{"bin": label(edges[i], edges[i + 1]), "lo": edges[i], "hi": edges[i + 1], "count": counts[i]} for i in range(bins)]


def read_jsonl(path: Path | str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class Result:
    """Collects what the UI renders. Every method returns self so calls chain.

    metric(key, label, value, fmt="num"|"int"|"pct"|"ms"|"bytes"|"text", accent="raw"|"kept"|"dup"|"hold"|"sky", help=...)
    chart(id, title, data, x, series=[{key,label,color?,form?}], type=..., ...)
        type: "bar" (compare magnitude) · "stacked" (parts of a whole across a
        dimension) · "line" (trend) · "area" · "scatter" (one number against
        another) · "donut" (parts of one whole, up to 5 slices and the rest folded
        into "other"; the first series is the value and x names each slice).
        A series may carry form="line"|"scatter"|"bar"|"area" to draw itself
        differently from the rest — measured points over a fitted line, say.
        Leave colour off unless it means something: the chart assigns colours that
        stay apart for colour-blind readers, and a donut uses one hue stepped
        light to dark.
    table(id, title, columns=[{key,label,fmt?}], rows=[...])
    tokens(id, title, tokens=[{text,id,kind?}])
    artifact(path, label)   output(key, value)   note(markdown)
    """

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.charts: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.token_views: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.outputs: dict[str, Any] = {}
        self.notes: list[str] = []
        self.started = time.time()

    def metric(self, key: str, label: str, value: Any, fmt: str = "num", accent: Optional[str] = None, help: Optional[str] = None) -> "Result":
        self.metrics.append({"key": key, "label": label, "value": value, "fmt": fmt, "accent": accent, "help": help})
        return self

    def chart(self, id: str, title: str, data: list[dict[str, Any]], x: str, series: list[dict[str, Any]], type: str = "bar", x_log: bool = False, y_log: bool = False, ref_x: Any = None, ref_label: Optional[str] = None, note: Optional[str] = None, stretch: bool = False, y_label: Optional[str] = None, x_label: Optional[str] = None, y_domain: Optional[list[float]] = None) -> "Result":
        self.charts.append({"id": id, "title": title, "type": type, "x": x, "series": series, "data": data, "x_log": x_log, "y_log": y_log, "ref_x": ref_x, "ref_label": ref_label, "note": note, "stretch": stretch, "y_label": y_label, "x_label": x_label, "y_domain": y_domain})
        return self

    def table(self, id: str, title: str, columns: list[dict[str, Any]], rows: list[dict[str, Any]], note: Optional[str] = None) -> "Result":
        self.tables.append({"id": id, "title": title, "columns": columns, "rows": rows, "note": note})
        return self

    def tokens(self, id: str, title: str, tokens: list[dict[str, Any]], note: Optional[str] = None) -> "Result":
        self.token_views.append({"id": id, "title": title, "tokens": tokens, "note": note})
        return self

    def artifact(self, path: str | Path, label: str) -> "Result":
        p = Path(path)
        rel = p.relative_to(_run_dir()).as_posix() if p.is_absolute() and _run_dir() in p.parents else str(path)
        size = p.stat().st_size if p.exists() else 0
        self.artifacts.append({"path": rel, "label": label, "bytes": size})
        return self

    def output(self, key: str, value: Any) -> "Result":
        self.outputs[key] = str(value) if isinstance(value, Path) else value
        return self

    def note(self, markdown: str) -> "Result":
        self.notes.append(markdown)
        return self

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (_run_dir() / "result.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"metrics": self.metrics, "charts": self.charts, "tables": self.tables, "tokens": self.token_views, "artifacts": self.artifacts, "outputs": self.outputs, "notes": self.notes, "elapsed_sec": round(time.time() - self.started, 3)}
        path.write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[atelier] result written to {path}", file=sys.stderr, flush=True)
        return path
