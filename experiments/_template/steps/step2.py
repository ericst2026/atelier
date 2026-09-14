"""Every step: read params + inputs, do work, report progress, write result.json."""
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress

parse_args()
P = params({"n": 10})
I = inputs()  # outputs of the previous step (+ materials_dir, workspace_dir, experiment_dir)
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
for i in range(int(P["n"])):
    progress(100 * i / int(P["n"]), f"item {i}", step=i, value=i * i)  # extra keywords become live series
R = Result()
R.metric("n", "Items", int(P["n"]), "int", "kept")
R.chart("squares", "Squares", [{"i": i, "sq": i * i} for i in range(int(P["n"]))], "i", [{"key": "sq", "label": "i²", "color": "sky"}], "line")
R.output("n", int(P["n"]))  # handed to the next step as inputs["n"]
R.save()
