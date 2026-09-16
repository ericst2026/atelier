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
PARAM_TYPES = {"int", "float", "bool", "select", "multiselect", "text", "textarea", "run", "material"}
MATERIAL_KINDS = {"model", "dataset"}


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
    # a step a student may re-implement: the form then offers the standard script
    # or their own file, and the run executes whichever they picked
    own_code: bool = False
    # a drawing that explains the step, shown large on the wall: a path inside the
    # experiment folder, e.g. figures/step1.svg
    figure: str = ""
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
    category: str = ""
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
            "category": self.category,
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
                    "own_code": s.own_code,
                    "figure": s.figure,
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
        # a material param lists prepared models or datasets under materials/
        # (kind: model|dataset, optional formats: [atelier, hf] or schemas: [documents, qa, pairs])
        if p.get("type") == "material" and p.get("kind") not in MATERIAL_KINDS:
            raise ValueError(f"{slug}/{step_id}/{key}: a material param needs kind: model or kind: dataset")
        # show_if: {other_key: value or [values]} hides the field unless the other param matches
        if "show_if" in p and not isinstance(p["show_if"], dict):
            raise ValueError(f"{slug}/{step_id}/{key}: show_if must be a mapping of param key to value")


def _validate_materials(slug: str, mats: list[dict[str, Any]], material_params: list[dict[str, Any]]) -> None:
    """materials: [{key, name, path, kind, optional, for: [param keys], description}].
    A material param can only be set to one of these, so a typo here would silently
    hide a choice; catch it when the experiment loads instead."""
    param_keys = {p["key"] for p in material_params}
    for m in mats:
        where = f"{slug}/materials/{m.get('key') or m.get('path')}"
        path = str(m.get("path") or "")
        if not path:
            raise ValueError(f"{where}: needs a path under materials/")
        kind = m.get("kind")
        if kind in MATERIAL_KINDS and path.replace("\\", "/").strip("/").split("/")[0] != f"{kind}s":
            raise ValueError(f"{where}: a {kind} lives under materials/{kind}s/, not {path!r}")
        unknown = set(m.get("for") or []) - param_keys
        if unknown:
            raise ValueError(f"{where}: 'for' names {sorted(unknown)}, which are not material params of this experiment")


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
                own_code=bool(s.get("own_code", False)),
                figure=str(s.get("figure") or ""),
                figures=s.get("figures") or [],
            )
        )
    mats = list(raw.get("materials") or [])
    _validate_materials(slug, mats, [p for s in steps for p in s.params if p.get("type") == "material"])
    return ExperimentSpec(
        slug=slug,
        dir=path,
        title=raw.get("title", slug),
        summary=raw.get("summary", ""),
        description=raw.get("description", ""),
        order=int(raw.get("order", 100)),
        tags=list(raw.get("tags") or []),
        difficulty=raw.get("difficulty", "intro"),
        category=str(raw.get("category") or ""),
        steps=steps,
        materials=mats,
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
        self._categories: list[dict[str, Any]] = []
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
        categories = self._load_categories(errors)
        by_slug = {slug: c["id"] for c in categories for slug in c.get("experiments") or []}
        for spec in specs.values():
            # experiment.yaml's own category wins; then categories.yaml; then Other
            spec.category = spec.category or by_slug.get(spec.slug, "other")
        self._specs, self._errors, self._categories, self._loaded_at = specs, errors, categories, time.time()

    def _load_categories(self, errors: dict[str, str]) -> list[dict[str, Any]]:
        path = self.root / "categories.yaml"
        if not path.exists():
            return []
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            out = []
            for c in raw.get("categories") or []:
                if not isinstance(c, dict) or not c.get("id"):
                    raise ValueError("every category needs an id")
                out.append({"id": str(c["id"]), "title": c.get("title") or str(c["id"]), "summary": c.get("summary", ""), "experiments": list(c.get("experiments") or [])})
            return out
        except Exception as exc:
            errors["categories.yaml"] = str(exc)
            return []

    def categories(self) -> list[dict[str, Any]]:
        """Sections for the experiments page, in file order, without the slug lists."""
        self.reload()
        return [{k: v for k, v in c.items() if k != "experiments"} for c in self._categories]

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
