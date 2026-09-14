"""Step 4 — a small student taught by the full model."""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.data import sft_batch
from atelier_mini.gen import generate
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import cosine_lr
from atelier_world import World

parse_args()
P = params({"preset": "tiny", "examples": 12000, "epochs": 3.0, "temperature": 2.0, "alpha": 0.7, "lr": 6e-4, "compare_scratch": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=88)
tok = MiniTokenizer.load(I["tokenizer"])
teacher, tck = MiniLM.load(I["model"], device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
system = I.get("system") or tck.get("system") or world.system_prompt
rows = list(world.instructions(int(P["examples"]), with_steps=True))
tasks = world.eval_set(200, seed=333_999)


def accuracy(model):
    gens = generate(model, tok, [t["prompt"] for t in tasks], 160, 0.0, batch_size=32, system=system)
    return sum(world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)) / len(tasks)


teacher_acc = accuracy(teacher)
progress(8, f"teacher: {teacher.num_params():,} parameters, {teacher_acc:.1%}")
cfg = MiniConfig.preset(P["preset"], teacher.config.vocab_size)
cfg.block_size = teacher.config.block_size
batch_size, T_temp, alpha = 24, float(P["temperature"]), float(P["alpha"])
steps_total = max(1, int(len(rows) * float(P["epochs"]) / batch_size))


def train_student(with_teacher: bool, tag: str, offset: float):
    torch.manual_seed(1)
    student = MiniLM(cfg).to(device)
    opt = student.optimizers(0.1, float(P["lr"]))
    history, cursor, t0 = [], 0, time.time()
    student.train()
    for it in range(steps_total):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, steps_total, float(P["lr"]), max(10, steps_total // 20))
        if cursor + batch_size > len(rows):
            cursor = 0
        batch = rows[cursor : cursor + batch_size]
        cursor += batch_size
        x, y, mask = sft_batch(batch, tok, cfg.block_size, device, system)
        logits, hard = student(x, y, mask)
        loss = hard
        if with_teacher:
            with torch.no_grad():
                t_logits, _ = teacher(x)
            m = mask.unsqueeze(-1).float()
            soft = F.kl_div(
                F.log_softmax(logits.float() / T_temp, dim=-1),
                F.softmax(t_logits.float() / T_temp, dim=-1),
                reduction="none",
            ).sum(-1)
            soft = (soft * mask.float()).sum() / mask.float().sum().clamp(min=1) * (T_temp**2)
            loss = alpha * soft + (1 - alpha) * hard
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it % 20 == 0:
            history.append({"step": it, "loss": float(loss)})
            progress(offset + 35 * it / steps_total, f"{tag} · step {it}/{steps_total} · loss {float(loss):.3f}", step=it, **{tag: float(loss)})
    return student, history, time.time() - t0


student, hist_d, secs_d = train_student(True, "distilled", 12)
acc_d = accuracy(student)
student.save(run_dir / "student.pt", {"distilled": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "teacher": I["model"]})
results = [{"model": "teacher", "params": teacher.num_params(), "accuracy": teacher_acc, "seconds": 0}, {"model": "distilled student", "params": student.num_params(), "accuracy": acc_d, "seconds": secs_d}]
hist_s = []
if bool(P["compare_scratch"]):
    scratch, hist_s, secs_s = train_student(False, "from scratch", 52)
    acc_s = accuracy(scratch)
    results.append({"model": "student from scratch", "params": scratch.num_params(), "accuracy": acc_s, "seconds": secs_s})
    del scratch
    torch.cuda.empty_cache()

curve = {}
for h in hist_d:
    curve.setdefault(h["step"], {"step": h["step"]})["distilled"] = h["loss"]
for h in hist_s:
    curve.setdefault(h["step"], {"step": h["step"]})["from scratch"] = h["loss"]

R = Result()
R.metric("student_accuracy", "Distilled student", acc_d, "pct", "kept", help=f"{student.num_params():,} parameters")
R.metric("teacher_accuracy", "Teacher", teacher_acc, "pct", "raw", help=f"{teacher.num_params():,} parameters")
R.metric("retained", "Share of the teacher retained", acc_d / max(teacher_acc, 1e-9), "pct", "sky")
R.metric("compression", "Size ratio", teacher.num_params() / max(student.num_params(), 1), "num", "hold", help="parameters in the teacher per parameter in the student")
R.chart("accuracy", "Accuracy", [{"model": r["model"], "accuracy": r["accuracy"]} for r in results], "model", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1], note="The gap between the two students is what the teacher's full distribution was worth, over and above the correct token alone.")
if hist_s:
    R.chart("loss", "Training loss", [curve[k] for k in sorted(curve)], "step", [{"key": "distilled", "label": "With a teacher", "color": "kept"}, {"key": "from scratch", "label": "Without", "color": "raw"}], "line", note="The two losses are not comparable in absolute terms — one includes a KL term — but their shapes are.")
R.table("results", "Results", [{"key": "model", "label": "Model"}, {"key": "params", "label": "Parameters", "fmt": "int"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "seconds", "label": "Training seconds", "fmt": "num"}], results)
R.artifact(run_dir / "student.pt", "student.pt")
R.output("student", str(run_dir / "student.pt")).output("student_accuracy", acc_d).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
R.save()
