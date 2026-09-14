"""python grader/grade.py --project <dir> --run-dir <dir>  → result.json with a `score` metric (0–100)."""
import os
from pathlib import Path

from atelier_sdk import Result, parse_args

args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", "."))
ok = (project / "project" / "main.py").exists()
R = Result()
R.metric("score", "Score", 100 if ok else 0, "num", "kept" if ok else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "detail", "label": "Detail"}], [{"name": "entry", "passed": ok, "detail": "project/main.py exists"}])
R.output("score", 100 if ok else 0)
R.save()
