"""Dependency-free checks of the pieces that do not need FastAPI.

    cd backend && python tests/test_core.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ATELIER_ROOT", tempfile.mkdtemp())

from atelier import registry, storage  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


# --- registry ---------------------------------------------------------------
root = Path(__file__).resolve().parents[2] / "experiments"
reg = registry.Registry(root)
reg.reload(force=True)
specs = reg.list()
check("registry loads the shipped experiments", len(specs) >= 5, f"{[s.slug for s in specs]}")
check("registry reports no manifest errors", not reg.errors, str(reg.errors))
check("every experiment has exactly four steps", all(len(s.steps) == 4 for s in specs))
check("step scripts exist", all((s.dir / st.script).exists() for s in specs for st in s.steps))
check("graders exist", all((s.dir / s.grader["script"]).exists() for s in specs if s.grader.get("script")))
check("sample projects exist", all(s.sample_dir.exists() for s in specs))
check("_template is ignored", all(s.slug != "my-experiment" for s in specs))

with tempfile.TemporaryDirectory() as tmp:
    bad = Path(tmp) / "bad"
    bad.mkdir()
    (bad / "experiment.yaml").write_text("slug: bad\ntitle: Bad\nsteps: [{id: one, title: One}]\n")
    r2 = registry.Registry(Path(tmp))
    r2.reload(force=True)
    check("a three-step experiment is rejected", "bad" in r2.errors, str(r2.errors))

# run params referencing other experiments must point at something real
slugs = {s.slug for s in specs}
refs_ok = True
for s in specs:
    for st in s.steps:
        for p in st.params:
            if p.get("type") == "run" and p.get("experiment") and p["experiment"] not in slugs:
                refs_ok = False
check("cross-experiment run params point at real experiments", refs_ok)

# --- storage ----------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    r = Path(tmp)
    (r / "a" / "b").mkdir(parents=True)
    (r / "a" / "b" / "f.txt").write_text("hello")
    (r / "a" / "__pycache__").mkdir()
    (r / "a" / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
    check("safe_join stays inside the root", storage.safe_join(r, "a/b/f.txt").exists())
    try:
        storage.safe_join(r, "../../etc/passwd")
        check("safe_join rejects traversal", False)
    except storage.StorageError:
        check("safe_join rejects traversal", True)
    entries = storage.tree(r)
    check("tree skips __pycache__", all("__pycache__" not in e["path"] for e in entries), str(entries))
    check("read_text returns the content", storage.read_text(r / "a" / "b" / "f.txt")["content"] == "hello")
    z = storage.zip_dir(r)
    check("zip_dir produces an archive", z[:2] == b"PK" and len(z) > 0)
    out = Path(tmp) / "restored"
    storage.unzip_to(z, out, 10 * 1024 * 1024)
    check("unzip_to round-trips", (out / "a" / "b" / "f.txt").read_text() == "hello")
    check("sha256_dir is stable", storage.sha256_dir(r) == storage.sha256_dir(r))
    log = r / "run.log"
    log.write_text("one\ntwo\nthree\n")
    first = storage.tail_log(log, 0)
    check("tail_log reads from an offset", first["text"].startswith("one") and first["eof"])
    check("tail_log continues", storage.tail_log(log, first["next_offset"])["text"] == "")
    check("last_lines returns the tail", storage.last_lines(log, 2) == ["two", "three"])

# --- the runner's progress protocol ------------------------------------------
# The runner pulls in redis and sqlalchemy; when they are not installed (a plain
# checkout without the backend requirements) this section is skipped rather than
# failing, since the rest of the checks are still worth running.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "_lib"))
try:
    from atelier.worker.runner import PROGRESS_PREFIX, LiveSeries
except ModuleNotFoundError as exc:
    print(f"[SKIP] runner checks — {exc}. Install backend/requirements.txt to run them.")
    LiveSeries = None

if LiveSeries is not None:
  with tempfile.TemporaryDirectory() as tmp:
    live = LiveSeries(Path(tmp) / "live.json")
    for line in ['{"pct": 10, "msg": "a", "series": {"step": 1, "loss": 2.0}}', '{"series": {"step": 2, "loss": 1.5}}']:
        payload = json.loads(line)
        live.add(dict(payload.get("series") or {}))
    live.flush()
    data = json.loads((Path(tmp) / "live.json").read_text())
    check("live series accumulate points", data["series"]["loss"] == [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.5}], json.dumps(data))
  check("progress prefix is what the SDK emits", PROGRESS_PREFIX == "::progress ")

# --- the SDK's Result document ------------------------------------------------
from atelier_sdk import Result, hist  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    R = Result()
    R.metric("k", "K", 1, "int", "kept").chart("c", "C", [{"x": 1, "y": 2}], "x", [{"key": "y", "label": "Y"}], "line").output("o", 5)
    doc = json.loads(R.save(Path(tmp) / "result.json").read_text())
    check("Result writes metrics, charts and outputs", doc["metrics"][0]["key"] == "k" and doc["charts"][0]["id"] == "c" and doc["outputs"]["o"] == 5)
h = hist([1, 2, 2, 3, 9], bins=3)
check("hist buckets values", sum(b["count"] for b in h) == 5, json.dumps(h))

print()
if failures:
    print(f"{len(failures)} failed: {failures}")
    sys.exit(1)
print("all checks passed")
