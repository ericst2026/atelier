"""Experiment registry: every folder under experiments/ with an experiment.yaml.

The registry is file based on purpose — teachers add or edit experiments by
editing files on the server, no database migration involved. Every experiment
must define exactly four steps."""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

STEP_COUNT = 4
PARAM_TYPES = {"int", "float", "bool", "select", "multiselect", "text", "textarea", "run"}


@dataclass
class StepSpec:
    index: int
    id: str
    title: str
    summary: str = ""
    description: str = ""
    script: str = ""
    gpus: int = 0
    timeout_min: int = 120
    needs_previous: bool = True
    params: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)  # which metrics to show on the rail

    def defaults(self) -> dict[str, Any]:
        return {p["key"]: p.get("default") for p in self.params if "key" in p}


@dataclass
class ExperimentSpec:
    slug: str
    dir: Path
    title: str
    summary: str = ""
    description: str = ""
    order: int = 100
    tags: list[str] = field(default_factory=list)
    difficulty: str = "intro"
    steps: list[StepSpec] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    project: dict[str, Any] = field(default_factory=dict)
    grader: dict[str, Any] = field(default_factory=dict)
    leaderboard: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_dir(self) -> Path:
        return self.dir / self.project.get("sample_dir", "sample")

    def step(self, n: int) -> StepSpec:
        if n < 1 or n > len(self.steps):
            raise KeyError(f"step {n} does not exist")
        return self.steps[n - 1]

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "slug": self.slug,
            "title": self.title,
            "summary": self.summary,
            "order": self.order,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "gpus": max([s.gpus for s in self.steps] + [0]),
            "steps": [
                {"index": s.index, "id": s.id, "title": s.title, "summary": s.summary, "gpus": s.gpus}
                for s in self.steps
            ],
            "leaderboard": self.leaderboard,
        }
        if full:
            d["description"] = self.description
            d["materials"] = self.materials
            d["project"] = {k: v for k, v in self.project.items() if k != "sample_dir"}
            d["grader"] = {k: v for k, v in self.grader.items() if k != "script"}
            d["steps"] = [
                {
                    "index": s.index,
                    "id": s.id,
                    "title": s.title,
                    "summary": s.summary,
                    "description": s.description,
                    "gpus": s.gpus,
                    "timeout_min": s.timeout_min,
                    "needs_previous": s.needs_previous,
                    "params": s.params,
                    "figures": s.figures,
                }
                for s in self.steps
            ]
        return d


def _validate_params(slug: str, step_id: str, params: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for p in params:
        key = p.get("key")
        if not key or key in seen:
            raise ValueError(f"{slug}/{step_id}: every param needs a unique 'key'")
        seen.add(key)
        if p.get("type", "text") not in PARAM_TYPES:
            raise ValueError(f"{slug}/{step_id}/{key}: unknown param type {p.get('type')!r}")


def load_experiment(path: Path) -> ExperimentSpec:
    raw = yaml.safe_load((path / "experiment.yaml").read_text(encoding="utf-8")) or {}
    slug = raw.get("slug") or path.name
    steps_raw = raw.get("steps") or []
    if len(steps_raw) != STEP_COUNT:
        raise ValueError(f"experiment {slug!r} must define exactly {STEP_COUNT} steps, found {len(steps_raw)}")
    steps = []
    for i, s in enumerate(steps_raw, start=1):
        params = s.get("params") or []
        _validate_params(slug, s.get("id", str(i)), params)
        steps.append(
            StepSpec(
                index=i,
                id=s.get("id", f"step{i}"),
                title=s.get("title", f"Step {i}"),
                summary=s.get("summary", ""),
                description=s.get("description", ""),
                script=s.get("script", f"steps/step{i}.py"),
                gpus=int(s.get("gpus", 0)),
                timeout_min=int(s.get("timeout_min", 120)),
                needs_previous=bool(s.get("needs_previous", i > 1)),
                params=params,
                figures=s.get("figures") or [],
            )
        )
    return ExperimentSpec(
        slug=slug,
        dir=path,
        title=raw.get("title", slug),
        summary=raw.get("summary", ""),
        description=raw.get("description", ""),
        order=int(raw.get("order", 100)),
        tags=list(raw.get("tags") or []),
        difficulty=raw.get("difficulty", "intro"),
        steps=steps,
        materials=list(raw.get("materials") or []),
        project=dict(raw.get("project") or {}),
        grader=dict(raw.get("grader") or {}),
        leaderboard=dict(raw.get("leaderboard") or {}),
    )


class Registry:
    def __init__(self, root: Path, reload_every: float = 5.0) -> None:
        self.root = Path(root)
        self.reload_every = reload_every
        self._specs: dict[str, ExperimentSpec] = {}
        self._errors: dict[str, str] = {}
        self._loaded_at = 0.0

    def reload(self, force: bool = False) -> None:
        if not force and time.time() - self._loaded_at < self.reload_every:
            return
        specs: dict[str, ExperimentSpec] = {}
        errors: dict[str, str] = {}
        if self.root.exists():
            for child in sorted(self.root.iterdir()):
                if child.name.startswith("_") or not (child / "experiment.yaml").exists():
                    continue
                try:
                    spec = load_experiment(child)
                    specs[spec.slug] = spec
                except Exception as exc:  # keep the platform up even if one manifest is broken
                    errors[child.name] = str(exc)
        self._specs, self._errors, self._loaded_at = specs, errors, time.time()

    @property
    def errors(self) -> dict[str, str]:
        self.reload()
        return dict(self._errors)

    def list(self) -> list[ExperimentSpec]:
        self.reload()
        return sorted(self._specs.values(), key=lambda s: (s.order, s.slug))

    def get(self, slug: str) -> ExperimentSpec:
        self.reload()
        if slug not in self._specs:
            raise KeyError(slug)
        return self._specs[slug]
