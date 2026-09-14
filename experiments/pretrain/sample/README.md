# Train a language model from scratch

Budget: **≤ 30M parameters**, one A6000, and whatever tokens you can push through
in your time slot. The grader loads `outputs/ckpt.pt`, checks the parameter
budget, and measures validation loss on a fixed TinyStories slice tokenized with
GPT-2 BPE (the same `train.bin`/`val.bin` format the guided steps produce).

- `project/model.py` — start from the reference GPT; try RMSNorm, rotary
  embeddings, SwiGLU, a different depth/width ratio, no weight tying …
- `project/train.py` — the loop is yours: schedule, batch size, sequence length,
  data mixing. It reads the prepared data with `--data <dir containing train.bin val.bin meta.json>`;
  by default it looks for the latest Data step run in your runs (pass the path shown on the Data step).
- Save `{"model_state", "config"}` to `outputs/ckpt.pt`. Keep `build_model(config)` working.

Leaderboard metric: validation loss (lower is better).
