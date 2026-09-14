# The whole course

Eighteen experiments, four steps each, in the order a model is actually built.
All of them are written, with a sample project and an automatic grader each.

Two tracks run in parallel through the same syllabus:

- **`world-*`** — the corpus is generated and the model trained from scratch. Nothing
  downloaded, everything checkable, models from 10M to 200M parameters.
- **`hf-*`** — the same idea on public data and small public models (135M–1.5B).
  Slower to run, closer to what students will meet outside the classroom.

Teach either, or pair them: run the `world` version to understand the mechanism,
then the `hf` version to see it at a scale someone else built.

| # | Experiment | The four steps | Needs | GPU | Status |
|---|---|---|---|---|---|
| **A. Data** ||||||
| 1 | **Tokenizer** | Corpus → Dedup → Train → Coverage | — | CPU | built |
| 2 | **Data curation** | Survey → Filter → Deduplicate → Mix | fineweb-edu, c4-raw | 1 | built |
| **B. Pretraining** ||||||
| 3 | **Pretraining** | Pack → Size → Train → Evaluate | — | 1–8 | built |
| 4 | **Scaling laws** | Plan → Sweep → Fit → Verify | — (pythia suite to compare) | 1–8 | built |
| 5 | **Distributed training** | Profile → Scale → Optimise → Verify | — | 2–8 | built |
| **C. Adaptation** ||||||
| 6 | **Fine-tuning** | Data → Baseline → Train → Evaluate | sft group | 1 | built |
| 7 | **Parameter-efficient tuning** | Setup → Sweep → Compare → Merge | Qwen2.5-0.5B, sft group | 1 | built |
| 8 | **Preference optimisation** | Pairs → Reference → Train → Evaluate | — | 1 | built |
| 9 | **Reinforcement learning** | Reward → Prompts → Train → Evaluate | gsm8k | 1 | built |
| **D. Capability** ||||||
| 10 | **Reasoning** | Prompting → Collect → Train → Evaluate | gsm8k, math-500 | 1 | built |
| 11 | **Retrieval** | Corpus → Encode → Train → Evaluate | — | 1 | built |
| 12 | **Retrieval on Wikipedia** | Index → Retrieve → Generate → Evaluate | wikipedia, squad, hotpotqa, embedders | 1 | built |
| 14 | **Tool use** | Tools → Traces → Train → Evaluate | — | 1 | built |
| 15 | **Long context** | Measure → Extend → Train → Evaluate | — | 1–2 | built |
| **E. Measuring and shipping** ||||||
| 13 | **Evaluation** | Tasks → Harness → Sensitivity → Report | eval group | 1 | built |
| 16 | **Efficiency** | Measure → Quantise → Accelerate → Distil | — | 1 | built |
| 17 | **Safety** | Policy → Measure → Train → Evaluate | pku-saferlhf, no-robots | 1 | built |

| 18 | **Interpretability** | Probe → Attention → Logit lens → Steering | pythia + checkpoints | 1 | built |

Seventeen, then. The last one is an elective: it teaches nothing about building a
model and everything about what one is.

## What each of them does

**2 · Data curation.** Two slices of the same web — one already filtered, one raw.
Students write heuristic filters (length, symbol ratio, repetition), then a
classifier filter, then dedupe and check for evaluation contamination, then choose
domain weights for the mix. Scored by the validation loss of a fixed small model
trained on their mix: the first experiment where data work is measured in loss.

**4 · Scaling laws.** Train four or five sizes on the same data with the same
recipe, fit loss against parameters and tokens, then predict the loss of a size
not yet trained and check the prediction. Fifteen minutes of GPU per point at the
tiny end, so a class can produce a real curve in one session.

**5 · Distributed training.** Profile one GPU, then run the same job on two, four
and eight with DDP; measure scaling efficiency and find where it breaks. Then
gradient accumulation, activation checkpointing and precision, and the same curve
again. The one experiment that needs the whole node, so it works well as a
demonstration with the class watching display 2.

**7 · Parameter-efficient tuning.** LoRA against full fine-tuning at equal
accuracy: how many trainable parameters, how much memory, how long. A rank sweep
from 2 to 128, then merging the adapter and confirming the merged model matches.

**8 · Preference optimisation.** Preference pairs rather than demonstrations. Build
the pairs, freeze a reference, train with DPO, and sweep β to watch the trade
between reward margin and drift from the reference. Runs directly against
experiment 9 — the same goal reached by two different mechanisms.

**11 · Retrieval.** No downloads: train an
embedder contrastively from the language model you already have. The corpus is
generated so that keyword search genuinely fails — documents come in clusters
sharing a place and an object, and every query is a paraphrase of its document, so
the words that match point at four documents and the words that would separate them
are not shared. BM25 gets about a third. In-batch negatives do the training, hard
negatives are mined from BM25's own mistakes, and the last step fuses the two
because they fail on different queries.

**12 · Retrieval on Wikipedia.** Chunk and embed a Wikipedia slice, build the index, measure
recall@k with two embedders and a reranker, then answer questions with and without
retrieval. Includes the question students always ask: what does a small model do
when the retrieved passage is wrong?

**14 · Tool use.** Define two or three tools (a calculator, a lookup, a search over
the index from experiment 11), generate or download traces, fine-tune on them, and
measure how often the model calls the right tool with valid arguments and uses
the result.

**15 · Long context.** Extend the context by scaling RoPE, continue training on
long documents, and evaluate with needle-in-a-haystack — a fact planted at a known
depth in a long document. The evaluation is generated, so the needle is never in
the training data.

**13 · Evaluation.** Build the harness rather than run one: likelihood scoring for
multiple choice, generation scoring for open answers, few-shot prompt assembly,
then a contamination check against the training corpus, and a model card. The
experiment that teaches students to distrust a leaderboard number.

**16 · Efficiency.** Measure latency and memory honestly, quantise to int8 and int4
and measure what accuracy costs, distil a small student from a larger teacher,
then KV caching, batching and speculative decoding, with a throughput curve.

**18 · Interpretability.** Linear probes at every layer to find where a property
becomes readable; induction heads located by feeding a repeated random sequence and
scoring where the attention lands; the logit lens, decoding the residual stream at
each layer through the output embedding to watch a prediction form; and steering by
a difference-of-means vector, which is the cheapest test of whether a direction
found by probing is actually used. Run the attention step across the Pythia
checkpoints and induction heads can be watched appearing part-way through training.

**17 · Safety.** Write a short policy, measure how often the model violates it,
train refusals with SFT or DPO, then red-team the result and measure the
over-refusal that training just introduced. The lesson is the trade-off: refusal
rate against helpfulness, both measured.

## Materials

`scripts/offline/materials.yaml` covers all of it: 23 models (all under 1.5B,
permissively licensed) and 34 dataset slices, about 27 GB in total.

```bash
python scripts/offline/fetch-materials.py --plan            # sizes, nothing downloaded
python scripts/offline/fetch-materials.py --tracks core     # 13 GB, what most classes need
python scripts/offline/fetch-materials.py                   # everything
python scripts/offline/fetch-materials.py --groups rag eval # one experiment's data
```

Tracks: `core` (13 GB) is the usual class set, `large` (10 GB) adds the 1.5B models
and the bigger corpora, `extended` (4 GB) covers the electives. Re-running skips
what is already complete, so a failed download resumes.

On Windows: `.\scripts\offline\fetch-materials.ps1 -Out D:\atelier-materials`.

When your own data and models are ready, the `world-*` track replaces the
downloads experiment by experiment — the page layout, the graders and the wall
displays do not change, only where the corpus comes from.
