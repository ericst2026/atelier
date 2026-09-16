"""A student's own version of one step.

Every step runs the script the experiment ships with. Where a step is marked
`own_code: true`, a student can instead write their own and run that: same form
params, same inputs from the previous step, same Result to save. Their file lives
in their workspace for the experiment, one per step:

    workspaces/<user id>/<experiment>/step<n>.py

The first time it is opened there is no file, so a prototype is generated from the
standard script — what it is given and what it must produce, with the body left to
write. Nothing is copied from the standard implementation itself: the point is to
write the step, not to edit an answer.
"""
import re
from pathlib import Path
from typing import Any, Optional

from . import storage
from .registry import ExperimentSpec, StepSpec

FILENAME = "step{n}.py"
# what the standard script reads and writes, recovered from its source
RE_INPUT = re.compile(r"""I(?:\.get\(|\[)["'](\w+)["']""")
RE_OUTPUT = re.compile(r"""R\.output\(\s*["'](\w+)["']""")
RE_METRIC = re.compile(r"""R\.metric\(\s*["'](\w+)["']\s*,\s*["']([^"']+)["']""")


def code_path(user_id: int, slug: str, step_no: int) -> Path:
    return storage.workspace_dir(user_id, slug) / FILENAME.format(n=step_no)


def _standard_source(spec: ExperimentSpec, step: StepSpec) -> str:
    p = spec.dir / step.script
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _contract(spec: ExperimentSpec, step: StepSpec) -> dict[str, Any]:
    """What the step is handed and what the rest of the experiment expects back."""
    src = _standard_source(spec, step)
    seen: dict[str, str] = {}
    for key, label in RE_METRIC.findall(src):
        seen.setdefault(key, label)
    return {
        "params": [{"key": p["key"], "label": p.get("label", p["key"]), "default": p.get("default")} for p in step.params if p.get("key")],
        "inputs": sorted(set(RE_INPUT.findall(src))),
        "outputs": sorted(set(RE_OUTPUT.findall(src))),
        "metrics": [{"key": k, "label": v} for k, v in seen.items()],
    }


def prototype(spec: ExperimentSpec, step: StepSpec) -> str:
    """A file to start from: the interface filled in, the work left to do."""
    c = _contract(spec, step)
    defaults = ", ".join(f'"{p["key"]}": {p["default"]!r}' for p in c["params"]) or ""
    lines = [
        '"""' + f"{spec.title} · step {step.index} — {step.title}",
        "",
        (step.summary or "").strip(),
        "",
        "Your own version of this step. It is run exactly like the standard one: the",
        "params below come from the form, the inputs from your previous step's run, and",
        "what you save in the Result is what the next step reads.",
        '"""',
        "import os",
        "from pathlib import Path",
        "",
        "from atelier_sdk import Result, inputs, params, parse_args, progress",
        "",
        "parse_args()",
        f"P = params({{{defaults}}})",
        "I = inputs()",
        "R = Result()",
        "run_dir = Path(os.environ.get(\"ATELIER_RUN_DIR\", \".\"))",
        "",
    ]
    lines.append("# --- what the form gives you ----------------------------------------------")
    for p in c["params"] or [{"key": "(none)", "label": "this step takes no params"}]:
        lines.append(f'#   P["{p["key"]}"]{"" if p["key"] == "(none)" else ":"} {p["label"]}')
    lines.append("")
    lines.append("# --- what the previous step gives you --------------------------------------")
    if c["inputs"]:
        for k in c["inputs"]:
            lines.append(f'#   I.get("{k}")')
    else:
        lines.append("#   nothing — this step starts from the params alone")
    lines.append("#   I[\"materials_dir\"], I[\"workspace_dir\"], I[\"experiment_dir\"] are always there.")
    lines.append("")
    lines.append("# --- your work -------------------------------------------------------------")
    lines.append("progress(0, \"starting\")")
    lines.append("raise SystemExit(\"Nothing is implemented yet: write this step, or run it with the standard code.\")")
    lines.append("")
    lines.append("# --- what you have to leave behind -----------------------------------------")
    if c["metrics"]:
        lines.append("# The figures the step is judged on (the board and the step card read these):")
        for m in c["metrics"][:6]:
            lines.append(f'#   R.metric("{m["key"]}", "{m["label"]}", value)')
    if c["outputs"]:
        lines.append("# The next step reads these, so they have to keep their names:")
        for k in c["outputs"]:
            lines.append(f'#   R.output("{k}", ...)')
    lines.append("R.save()")
    return "\n".join(lines) + "\n"


def read(user_id: int, spec: ExperimentSpec, step: StepSpec) -> dict[str, Any]:
    """The student's file, or the prototype when they have not started."""
    p = code_path(user_id, spec.slug, step.index)
    if p.exists():
        return {"code": p.read_text(encoding="utf-8", errors="replace"), "saved": True, "path": str(p)}
    return {"code": prototype(spec, step), "saved": False, "path": str(p)}


def write(user_id: int, spec: ExperimentSpec, step: StepSpec, code: str) -> Path:
    p = code_path(user_id, spec.slug, step.index)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code.replace("\r\n", "\n"), encoding="utf-8", newline="")
    return p


def exists(user_id: int, slug: str, step_no: int) -> bool:
    return code_path(user_id, slug, step_no).exists()


def reset(user_id: int, spec: ExperimentSpec, step: StepSpec) -> str:
    """Back to the prototype, losing what was written."""
    code = prototype(spec, step)
    write(user_id, spec, step, code)
    return code


def compile_error(code: str) -> Optional[str]:
    """A syntax error, before a run is queued for it."""
    try:
        compile(code, "step.py", "exec")
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return None
