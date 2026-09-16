"""A drawing of what a step does, made from the step itself.

The wall shows one of these beside the explanation. Rather than sixty-four hand-drawn
pictures that drift out of date the moment a step changes, the diagram is built from
the step's own contract: what it is handed, what it does, what it leaves behind, and
the figures it is judged on. An experiment that wants a real drawing sets
`figure: figures/step1.svg` on the step and that is served instead.

It is read from across a room, so: few boxes, big type, one row of flow.
"""
from typing import Any, Optional

from . import stepcode
from .registry import ExperimentSpec, StepSpec

# the app's own palette, checked for colour-blind separation
INK = "#e9eef7"
MUTED = "#93a3bf"
FAINT = "#5f7192"
PANEL = "#15243b"
INSET = "#0b1526"
LINE = "#2a3d5f"
RAW = "#f4a259"
KEPT = "#5fd3b8"
SKY = "#7cc4ff"

W, H = 1200, 520


def _esc(t: str) -> str:
    return (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _human(key: str) -> str:
    """A name a room can read: the scripts see tokenizer_run_run, people see a
    tokenizer run."""
    name = key.strip()
    while name.endswith("_run"):
        name = name[: -len("_run")]
    return f"{name.replace('_', ' ')} run" if key.endswith("_run") else name.replace("_", " ")


def _wrap(text: str, width: int, limit: int = 3) -> list[str]:
    """Break a label into at most `limit` lines of about `width` characters."""
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
        if len(lines) == limit:
            break
    if cur and len(lines) < limit:
        lines.append(cur)
    if len(lines) == limit and len(" ".join(lines)) < len(text or ""):
        lines[-1] = lines[-1][: width - 1] + "…"
    return lines or [""]


def _box(x: int, y: int, w: int, h: int, title: str, lines: list[str], accent: str, big: bool = False) -> str:
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{PANEL if big else INSET}" stroke="{accent}" stroke-width="{3 if big else 2}"/>',
        f'<text x="{x + w // 2}" y="{y + (44 if big else 34)}" text-anchor="middle" fill="{accent}" font-size="{26 if big else 20}" font-weight="700">{_esc(title)}</text>',
    ]
    ty = y + (82 if big else 64)
    for line in lines:
        out.append(f'<text x="{x + w // 2}" y="{ty}" text-anchor="middle" fill="{INK if big else MUTED}" font-size="{22 if big else 19}">{_esc(line)}</text>')
        ty += 28
    return "".join(out)


def _arrow(x1: int, x2: int, y: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 14}" y2="{y}" stroke="{FAINT}" stroke-width="3"/>'
        f'<polygon points="{x2},{y} {x2 - 16},{y - 9} {x2 - 16},{y + 9}" fill="{FAINT}"/>'
    )


def generate(spec: ExperimentSpec, step: StepSpec) -> str:
    """An SVG of this step: what comes in, what it does, what it leaves."""
    c = stepcode.contract(spec, step)
    prev = spec.step(step.index - 1) if step.index > 1 else None

    incoming: list[str] = []
    if prev is not None and step.needs_previous:
        incoming.append(f"step {prev.index}: {prev.title}")
    for key in (c.get("inputs") or [])[:3]:
        if key not in ("materials_dir", "workspace_dir", "experiment_dir", "parent_run_id", "parent_run_dir", "code_source", "class_session"):
            incoming.append(_human(key))
    if not incoming:
        incoming = ["the world, and what you set"]

    outgoing = [_human(k) for k in (c.get("outputs") or [])[:3]] or ["a result to read"]
    figures = [m["label"] for m in (c.get("metrics") or [])[:3]]
    settings = [p["label"] for p in (c.get("params") or [])[:3]]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">',
        f'<rect width="{W}" height="{H}" fill="none"/>',
        f'<text x="{W // 2}" y="46" text-anchor="middle" fill="{MUTED}" font-size="24">{_esc(spec.title)}</text>',
        f'<text x="{W // 2}" y="88" text-anchor="middle" fill="{INK}" font-size="34" font-weight="700">Step {step.index} · {_esc(step.title)}</text>',
    ]
    y = 140
    parts.append(_box(40, y, 300, 190, "it is given", _wrap(", ".join(incoming), 26, 3), SKY))
    parts.append(_box(450, y, 300, 190, f"step {step.index}", _wrap(step.summary or step.title, 24, 3), RAW, big=True))
    parts.append(_box(860, y, 300, 190, "it leaves", _wrap(", ".join(outgoing), 26, 3), KEPT))
    parts.append(_arrow(340, 450, y + 95))
    parts.append(_arrow(750, 860, y + 95))

    foot = y + 240
    if settings:
        parts.append(f'<text x="40" y="{foot}" fill="{FAINT}" font-size="20">you choose: {_esc(", ".join(settings))}</text>')
    if figures:
        parts.append(f'<text x="40" y="{foot + 34}" fill="{FAINT}" font-size="20">judged on: {_esc(", ".join(figures))}</text>')
    if step.own_code:
        parts.append(f'<text x="40" y="{foot + 68}" fill="{RAW}" font-size="20">you may write this step yourself</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_for(spec: ExperimentSpec, step: StepSpec) -> Optional[dict[str, Any]]:
    """Either the drawing the experiment ships, or one made from the step."""
    if step.figure:
        path = spec.dir / step.figure
        if path.exists():
            return {"path": path}
    return {"svg": generate(spec, step)}
