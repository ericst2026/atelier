"""Your policy, as code, and the data it implies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "safety"))
from lib_safety import default_policy, refusal_text  # noqa: E402


def should_refuse(prompt: str) -> bool:
    return default_policy(prompt) == "refuse"


def build_data(prompts: list[dict]) -> list[dict]:
    """prompts: [{"prompt", "expected", "source"}]. Return {prompt, target} pairs.
    Include helpful answers as well as refusals, or the model learns only to decline."""
    out = []
    for p in prompts:
        if should_refuse(p["prompt"]):
            out.append({"prompt": p["prompt"], "target": refusal_text(p["prompt"])})
    return out
