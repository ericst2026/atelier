"""Helpers for the GPU experiments (transformers / peft / trl), all offline.

Materials layout expected on the server:
    materials/models/<name>/                 HF snapshot (config.json, tokenizer.json, *.safetensors)
    materials/datasets/<group>/<name>/       train.jsonl [test.jsonl] (see scripts/offline/fetch-materials.py)
"""
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

MATERIALS = Path(os.environ.get("ATELIER_MATERIALS", "materials"))


# --- files -------------------------------------------------------------------
def read_jsonl(path: Path | str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def dataset_dir(group: str, name: str) -> Path:
    p = MATERIALS / "datasets" / group / name
    if not p.exists():
        raise FileNotFoundError(f"dataset {group}/{name} is not on this server ({p}). Run scripts/offline/fetch-materials.py on a connected machine and copy the folder.")
    return p


def dataset_split(group: str, name: str, split: str = "train", limit: Optional[int] = None) -> list[dict[str, Any]]:
    d = dataset_dir(group, name)
    for cand in (d / f"{split}.jsonl", d / f"{split}.json"):
        if cand.exists():
            return read_jsonl(cand, limit)
    raise FileNotFoundError(f"{d} has no {split}.jsonl")


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    m = MATERIALS / "models" / name_or_path
    if m.exists():
        return m
    raise FileNotFoundError(f"model {name_or_path!r} is not on this server (looked in {m}). Copy the HF snapshot there.")


# --- models ------------------------------------------------------------------
def torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(name, torch.bfloat16)


def load_tokenizer(path: str | Path, padding_side: str = "left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(path: str | Path, dtype: str = "bf16", device: Optional[str] = None, adapter: Optional[str | Path] = None):
    import torch
    from transformers import AutoModelForCausalLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch_dtype(dtype), local_files_only=True, trust_remote_code=False)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
    return model.to(device).eval()


def count_params(config_path: str | Path) -> dict[str, int]:
    """Parameter count without loading weights (meta device)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters() if "embed" in n or "lm_head" in n)
    return {"total": total, "embeddings": emb, "non_embedding": total - emb, "layers": getattr(cfg, "num_hidden_layers", 0), "hidden": getattr(cfg, "hidden_size", 0)}


def chat_prompt(tok, user: str, system: Optional[str] = None, assistant_prefix: str = "") -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix
    return (f"System: {system}\n\n" if system else "") + f"User: {user}\n\nAssistant:" + assistant_prefix


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0, batch_size: int = 16, num_return_sequences: int = 1, progress: Optional[Callable[[int, int], None]] = None) -> list[list[str]]:
    """Returns, for every prompt, a list of `num_return_sequences` completions (new tokens only)."""
    import torch

    out: list[list[str]] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5) if temperature > 0 else None, top_p=top_p if temperature > 0 else None, num_return_sequences=num_return_sequences, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for i in range(len(batch)):
            out.append(texts[i * num_return_sequences : (i + 1) * num_return_sequences])
        if progress:
            progress(min(len(prompts), start + len(batch)), len(prompts))
    return out


# --- math answers (GSM8K / MATH style) --------------------------------------
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_gold(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    return answer_field.strip()


def extract_answer(text: str) -> Optional[str]:
    """Last boxed / 'answer is' / '####' number, else the last number in the text."""
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip().replace(",", "")
    m = re.findall(r"####\s*([^\n]+)", text)
    if m:
        cand = NUM_RE.findall(m[-1])
        return cand[-1].replace(",", "") if cand else m[-1].strip()
    m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    if m:
        return m[-1].replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def answers_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    pred, gold = pred.strip().replace("$", ""), gold.strip().replace("$", "")
    if pred == gold:
        return True
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-6, abs_tol=1e-6)
    except ValueError:
        return False


def format_reward(text: str) -> float:
    """Small bonus for a clearly marked final answer."""
    return 0.2 if ("####" in text or "\\boxed" in text or re.search(r"answer is", text, re.I)) else 0.0


# --- training helpers --------------------------------------------------------
class ProgressCallback:
    """transformers TrainerCallback that turns trainer logs into ::progress lines."""

    def __init__(self, total_steps: int, label: str = "train"):
        from transformers import TrainerCallback

        cb = self

        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                logs = logs or {}
                series = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and (k in ("loss", "eval_loss", "learning_rate", "grad_norm") or k.startswith(("reward", "kl", "completion")))}
                series = {k.replace("/", "_"): v for k, v in series.items()}
                from atelier_sdk import progress

                progress(100.0 * state.global_step / max(1, total_steps), f"{label} step {state.global_step}/{total_steps}" + (f" · loss {logs['loss']:.3f}" if "loss" in logs else ""), step=state.global_step, **series)
                cb.history.append({"step": state.global_step, **logs})

        self.history: list[dict[str, Any]] = []
        self.callback = _CB()


def history_chart(history: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for h in history:
        row = {"step": h.get("step", 0)}
        for k in keys:
            if k in h and isinstance(h[k], (int, float)):
                row[k.replace("/", "_")] = h[k]
        if len(row) > 1:
            rows.append(row)
    return rows


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: Optional[list[str]] = None):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")


def sft_train(model_path: Path, train_rows: list[dict[str, Any]], out_dir: Path, val_rows: Optional[list[dict[str, Any]]] = None, method: str = "lora", lora: Optional[dict[str, Any]] = None, epochs: float = 1.0, lr: float = 2e-4, batch_size: int = 4, grad_accum: int = 4, max_length: int = 1024, packing: bool = False, warmup_ratio: float = 0.03, dtype: str = "bf16", gradient_checkpointing: bool = True, label: str = "sft") -> dict[str, Any]:
    """Supervised fine-tuning with TRL on rows shaped {"messages": [...]}. Returns training history."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    tok = load_tokenizer(model_path, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch_dtype(dtype), local_files_only=True)
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if val_rows else None
    steps_per_epoch = math.ceil(len(train_rows) / (batch_size * grad_accum * max(1, torch.cuda.device_count())))
    total = max(1, int(steps_per_epoch * epochs))
    cb = ProgressCallback(total, label)
    cfg = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        packing=packing,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=max(10, total // 8),
        save_strategy="no",
        bf16=dtype == "bf16" and torch.cuda.is_available(),
        fp16=dtype == "fp16" and torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",
        seed=1,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok, peft_config=lora_config(**(lora or {})) if method == "lora" else None, callbacks=[cb.callback])
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    save_dir = out_dir / ("adapter" if method == "lora" else "model")
    trainer.save_model(str(save_dir))
    tok.save_pretrained(str(save_dir))
    final_eval = trainer.evaluate() if val_ds is not None else {}
    return {"history": cb.history, "elapsed_sec": elapsed, "save_dir": str(save_dir), "method": method, "final_eval": final_eval, "total_steps": total}


def eval_loss(model, tok, rows: list[dict[str, Any]], max_length: int = 1024, batch_size: int = 4) -> float:
    """Mean token loss of assistant turns is expensive to mask exactly; this measures the whole rendered conversation."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) if getattr(tok, "chat_template", None) else "\n".join(m["content"] for m in r["messages"]) for r in rows]
    total, n = 0.0, 0
    tok.padding_side = "right"
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            out = model(**enc, labels=labels)
        ntok = int((labels != -100).sum())
        total += float(out.loss) * ntok
        n += ntok
    tok.padding_side = "left"
    return total / max(1, n)


def grpo_train(model_path: Path, rows: list[dict[str, Any]], reward_funcs: list[Callable], out_dir: Path, num_generations: int = 8, max_completion_length: int = 256, max_prompt_length: int = 512, temperature: float = 0.9, beta: float = 0.04, lr: float = 1e-6, max_steps: int = 200, prompts_per_step: int = 4, dtype: str = "bf16", label: str = "grpo") -> dict[str, Any]:
    """GRPO with TRL on rows shaped {"prompt": [messages], <extra columns passed to reward funcs>}."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    tok = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch_dtype(dtype), local_files_only=True)
    cb = ProgressCallback(max_steps, label)
    cfg = GRPOConfig(
        output_dir=str(out_dir / "trainer"),
        learning_rate=lr,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=prompts_per_step,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        temperature=temperature,
        beta=beta,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        bf16=dtype == "bf16" and torch.cuda.is_available(),
        report_to="none",
        seed=1,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok, callbacks=[cb.callback])
    t0 = time.time()
    trainer.train()
    save_dir = out_dir / "policy"
    trainer.save_model(str(save_dir))
    tok.save_pretrained(str(save_dir))
    return {"history": cb.history, "elapsed_sec": time.time() - t0, "save_dir": str(save_dir), "total_steps": max_steps}


def completion_text(c: Any) -> str:
    """GRPO hands conversational completions as [{"role","content"}]; plain ones as str."""
    if isinstance(c, list):
        return "".join(m.get("content", "") for m in c if isinstance(m, dict))
    return str(c)
