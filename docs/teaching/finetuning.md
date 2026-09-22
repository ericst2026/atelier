# Teaching guide: Fine-tuning (experiment 6)

A teacher's companion to the **Fine-tuning** experiment (`experiments/world-sft`). It covers:

- what the class learns;
- how to run the session on this platform;
- what to put on the wall at each point;
- sample classroom conversations with the questions students usually ask.

Everything here is taken from the experiment as it is built, and the numbers are from runs on this server.

---

## 1. At a glance

| | |
|---|---|
| **Big idea** | A pretrained model *continues* text. An assistant *answers* and stops. Supervised fine-tuning (SFT) closes that gap with demonstrations. |
| **Students will be able to** | explain what SFT changes and what it can't; build a demonstration set; read a training curve; measure improvement honestly against a baseline |
| **Steps** | 1 Data → 2 Baseline → 3 Train → 4 Evaluate |
| **Comes after** | Tokenizer (exp. 1) and Pretraining (exp. 5). Their runs are the usual starting point |
| **Leads into** | Parameter-efficient tuning (exp. 7, LoRA against full fine-tuning) and DPO / RL |
| **Length** | one 90-minute session on a GPU; two sessions, or shorter runs, on a CPU-only server (see §3) |
| **Own code** | steps 2, 3 and 4 can be re-implemented by students; step 1 can't |

### The one sentence to leave on the board

> Fine-tuning teaches the model **the shape of the answer** almost at once. Teaching it **the answer** is the hard part, and only a fair before-and-after measurement shows which one you got.

This platform's own runs show exactly that: after fine-tuning, **99.7 %** of the answers were written in the right format, but accuracy only rose from **5 % to 11 %** (§6, step 4).

---

## 2. Background the teacher needs

**Base model against assistant.** A base model is trained to predict the next token of web-like text. Ask it "What is 21 − 17 + 13?" and it may write three more questions, a story, or nothing useful. It has never learned that a question is a turn that expects an answer.

**Demonstrations.** SFT trains on pairs:

- a **prompt**: the instruction and the question;
- a **target**: what a good answer looks like.

Here the pairs are *generated* by the course's small "world". It is a deterministic generator of checkable problems in eight **task families**:

| family | example question | answer |
|---|---|---|
| arith | What is 21 − 17 + 13? | 17 |
| count | a list of places with numbers, then "Count the eggs." | 39 |
| sort | Put these in order, smallest first: 32, 33, 69, 53, 77. | 32, 33, 53, 69, 77 |
| lookup | three facts, then "Where does Milo keep a cotton sail?" | the harbour |
| path | a map, then "Yusuf starts at the observatory and goes south, north…" | the observatory |
| shop | "Anders buys 5 eggs at 2 coins each and 5 pears at 2 coins each…" | 20 |
| compare | "Bo has 17 baskets and Ximena has 12. How many more does Bo have?" | 5 |
| seq | What number comes next: 4, 12, 36, 108, 324, 972? | 2916 |

Every answer can be checked, so evaluation is **exact match**. There's no judge model and no arguing about what counts as good.

**The target format.** With *reasoning steps* switched on, a target looks like this:

```
First 21 - 17 = 4.
Then 4 + 13 = 17.
The answer is 17.
Answer: 17
```

The line starting `Answer:` is what the grader reads. The instruction the model gets (the "system prompt") is:

> Answer the question. Put the final answer on a line beginning with 'Answer:'.

**Loss masking.** For the class's own models (the mini-model path), the prompt tokens are **masked out of the loss**. The model is scored only on predicting the target, so it learns to answer, not to write more questions. The HuggingFace path works differently; see §9.

**Two ways the step trains, chosen by the base model:**

| base model | how it's trained |
|---|---|
| a class Pretraining run (mini model, `model.pt`) | **full fine-tuning**: every weight moves |
| a prepared HuggingFace model (SmolLM2, Qwen2.5, GPT-2, Pythia) | a **LoRA adapter** (rank 16, alpha 32, on the attention and MLP projections); the base weights stay frozen |

**Held-out questions.** Training uses generator seed 21 (the student can change it). Steps 2 and 4 grade questions from seed 555 003: the same generator, never the same problems. The project grader uses yet another seed (77 015 413), which students never see.

---

## 3. Before the class

### Checklist

- [ ] Every student has a finished **Tokenizer step 3** and **Pretraining step 3** run. If not, choose a prepared model on the class's start materials, e.g. SmolLM2-135M.
- [ ] Open the class for **Fine-tuning** on the Class page, and set the start materials (the base model) so everyone begins from the same place.
- [ ] Grant the experiment to anyone who isn't in the class but should see it.
- [ ] Check the wall: screens 1–4 follow steps 1–4, and screen 5 shows the hardware dashboard or the leaderboard.
- [ ] Decide the run sizes from the timings below, **before** students press Run.

### How long runs take

| run | GPU (estimate) | CPU-only server (1 slot, 3 GB), measured here where a number is given |
|---|---|---|
| Step 1 Data (20 000 demonstrations) | seconds | about a minute |
| Step 2 Baseline (300 questions × 3 formats) | 1–3 min | long; use 100 questions and one or two formats |
| Step 3 Train, mini model, defaults (2 epochs, batch 24 = **1 666 steps**) | minutes | a ~1 M-parameter base: **about 5 min** (measured: 322 s); the tiny preset (6.5 M): about 2 h, so use **0.25 epochs** (about 15 min) |
| Step 3 Train, SmolLM2-135M LoRA, defaults | minutes | about 18 h (hits the 3 h limit); use **0.05 epochs** (about 25 min) |
| Step 4 Evaluate (300 questions, base and tuned) | 1–3 min | long; use 100 questions |

On a CPU server, one run goes at a time, so a class queues behind itself. Two ways to keep a session moving:

1. **The front-of-room run.** You run each step once at the front with class-sized settings, and students watch it on the wall ("a student's run, live" on the step's screen). Students run their own copies afterwards or overnight.
2. **Small settings for everyone.** Give out the CPU numbers above. A 0.25-epoch run is enough to see the curves bend and the format appear.

If a run fails, the live view says why, for example "It was killed for using more memory than the machine allows", or that it ran past its time limit. Use that as a teaching moment, not a problem to hide.

---

## 4. Lesson plan

| time | segment | wall |
|---|---|---|
| 0–10 | Hook: what a base model does with a question | screen 2: "what the step is" |
| 10–25 | Step 1: build the demonstrations | screen 1: explanation, then a student's results |
| 25–40 | Step 2: the baseline, the number to beat | screen 2: a student's results |
| 40–65 | Step 3: train, and read the curves | screen 3: **a student's run, live** |
| 65–85 | Step 4: evaluate, then what changed and what didn't | screen 4: a student's results; screen 5: leaderboard |
| 85–90 | Exit ticket | — |

On a CPU server, stop after step 3 in the first session and start the second session with step 4.

---

## 5. The hook (10 min)

Put the baseline's "What it said" table on the wall. Use any finished step 2 run, or run one yourself before class.

> **Teacher:** Here's a model we pretrained ourselves. I asked it "What is 21 − 17 + 13?". Read what it said.
>
> **Student A:** It just… kept going. It wrote another question.
>
> **Teacher:** Right. Is it broken?
>
> **Student B:** No, it's doing what it was trained to do: guess the next bit of text. On the internet, a question is often followed by another question.
>
> **Teacher:** Exactly. Pretraining taught it language. Nobody taught it that a question expects an answer, or when to stop. Today we teach it that. Then we measure, honestly, how much it actually learned.

**Question to throw out:** *"What would you need to show a model so it learns to answer?"* Collect answers. You're steering toward: examples of questions paired with good answers. That's a demonstration dataset.

---

## 6. The four steps

### Step 1 — Data: generate demonstrations

**What students set:**

- the base model: their Pretraining run, or a prepared model;
- demonstrations from the world, or a prepared dataset;
- how many (default 20 000);
- **whether to include the reasoning steps**;
- which families, the difficulty, the held-out size and the seed.

**What to show:** a student's results. Point at the "What the model will be trained on" table, with its *Question (masked out of the loss)* and *Target (scored)* columns. The live chart during the run shows demonstrations being written, one line per family.

> **Teacher:** Look at the two columns. The model *reads* the question, but it's only *graded* on the target. Why would we hide the question from the loss?
>
> **Student C:** Otherwise it would also learn to write questions?
>
> **Teacher:** Yes. Every token we score is something we're asking it to produce. We want answers, not more questions.
>
> **Student D:** Then why give it the question at all?
>
> **Teacher:** Because the answer depends on it. It's *context*: the model sees it, it just isn't asked to predict it.

**The biggest single choice: with or without steps.**

> **Student A:** Why would anyone turn the steps off? They're longer and slower.
>
> **Teacher:** That's exactly the trade-off. Steps cost tokens and training time. Without them, the model has to jump from the question straight to "Answer: 17". A small model often can't do that for multi-step problems like shop or path. With steps, each line is a small, learnable move. This experiment says that choice usually moves accuracy more than any other setting on the page, and it's measurable: one run each way.

**Good question students ask:** *"Is it cheating that the data comes from the same generator as the test?"*

> **Teacher:** It's the same *kind* of problem, but never the same problem: the held-out questions use a different seed. That's the honest version of what every real fine-tune does: train on the task you care about, test on examples you didn't train on. What would be cheating is testing on the training rows, and the platform makes sure that can't happen.

**A discussion point (known quirk).** Show a **sort** demonstration with steps on:

```
Then 69, 77 = 53.
Then 77 = 69.
Then - = 77.
The answer is 32, 33, 53, 69, 77.
Answer: 32, 33, 53, 69, 77
```

The sort "reasoning" reuses the arithmetic sentence ("Then … = …"), so it reads as nonsense, and only its last four lines are kept, so the smallest numbers are never shown being picked.

> **Teacher:** Read the sort steps. Would *you* learn to sort from these?
>
> **Student B:** Not really. "Then 77 = 69" isn't even true.
>
> **Teacher:** So what do you predict for the sort family after training, with steps and without?
>
> **Student B:** Maybe without steps is better for sort?
>
> **Teacher:** Let's find out in step 4. It's a live example of the most important rule in SFT: **the model learns exactly what the demonstrations show, including their mistakes.**

**If a student picks a prepared dataset:** Alpaca and Dolly have free-text answers. The page itself warns that they're poor for exact-match grading. On this server, SmolLM2-135M on Alpaca-cleaned scored **0 %** in the baseline and will stay near 0 % after training, because a paragraph never matches a reference paragraph word for word. That's a great conversation about *how you measure*, not about the model. GSM8K has checkable numeric answers and works with exact match.

---

### Step 2 — Baseline: what the base model already does

**What it does:** asks the *untouched* base model the same held-out questions that step 4 uses, with three prompt formats:

- `bare`: the question alone;
- `system`: the instruction above it;
- `fewshot`: two solved examples first.

**What to show:** the "Accuracy by prompt format" chart. It has two bars per format: **Correct** and **Wrote an answer line**.

On this server a class-pretrained model scored **4.7 % (bare), 5.0 % (system) and 6.7 % (few-shot)**.

> **Teacher:** Why do we measure *before* we train?
>
> **Student C:** So we know how much the training added?
>
> **Teacher:** Yes. And look: the few-shot prompt already bought two points, without changing a single weight. If we didn't measure that, we'd give fine-tuning credit for something a better prompt already does.
>
> **Student D:** What's the "Wrote an answer line" bar?
>
> **Teacher:** How often it even *tried* to answer in our format. A base model often gets the format wrong before it gets the answer wrong. Remember that bar in step 4: it's the first thing fine-tuning fixes.

**Question to throw out:** *"If few-shot helps, why not just always use few-shot?"* It costs context on every single question, and it helps less than training does. Keep that in mind for step 4.

**Own code (step 2):** students can add a prompt format of their own, e.g. "Let's think step by step" or three shots instead of two, and hand it in. Their results sit beside the standard code's on the same settings, and you can put that comparison on the wall.

---

### Step 3 — Train: fine-tune with the prompt masked out

**What students set:**

- epochs (default 2);
- learning rate (default 2 × 10⁻⁴, well below pretraining's);
- examples per step (default 24);
- warmup and weight decay;
- whether to keep the instruction in the prompt.

**Number of optimizer steps** = demonstrations × epochs ÷ batch. At the defaults that's 20 000 × 2 ÷ 24 = **1 666 steps**.

**What to show: screen 3, "a student's run, live".** Pick a student who is running now. The wall shows their status, progress, time left against the time limit, the end of their log, and live charts:

| chart | what to say about it |
|---|---|
| **Loss**: train batch and validation | goes down fast, then flattens; **stop when validation turns upward** |
| **Gradient norm** (before clipping) | spikes early, then settles; a spike later means something went wrong (lr too high) |
| **Learning rate** | warmup ramp, then a cosine decay; it's the schedule, not something the model learns |

> **Teacher:** The loss dropped a lot in the first hundred steps, then almost stopped. What was learned in those first steps?
>
> **Student A:** The easy stuff?
>
> **Teacher:** The *format*. "Answer:" at the end, the "First… Then…" pattern. Those tokens appear in every target, so they're cheap to learn. The actual arithmetic is the slow tail.
>
> **Student B:** Why is our learning rate so much smaller than in pretraining?
>
> **Teacher:** The model already knows a lot. A big step would overwrite it. We want to nudge it toward answering, not rebuild it. Too high, and a small model "forgets how to write sentences": the experiment's own warning is that it starts answering everything with the most common number in the data.
>
> **Student C:** The validation loss is going up at the end of mine. Did I break it?
>
> **Teacher:** No, you found overfitting. It's memorising the demonstrations instead of learning the task. The fix is to train *less*: fewer epochs, or stop at the lowest point. More steps won't help.

**If a run fails, put its live view up.** It shows the error and the end of the log:

- "killed for using more memory than the machine allows": the batch didn't fit (the platform already splits batches on CPU, but a hand-written loop may not);
- "ran past its time limit": too many steps for this machine; lower the epochs.

**Own code (step 3):** ideas for students who re-implement it:

- freeze the lower layers and train only the top ones;
- a different learning-rate schedule;
- stop early at the best validation loss and keep that checkpoint;
- oversample the families that were weakest in the baseline.

---

### Step 4 — Evaluate: accuracy per family, before and after

**What it does:** greedy answers (temperature 0, up to 192 tokens) to the held-out questions, from the base model and from the fine-tuned model. Each is graded by exact match on the `Answer:` line.

**What to show:**

- the before-and-after bars;
- the "Accuracy by task family" chart;
- the two tables: **Questions fine-tuning fixed** and **Questions it lost**.

Results on this server (class-pretrained mini model, defaults):

| | before | after |
|---|---|---|
| Accuracy | 5.0 % | **11.0 %** (+6 points) |
| Wrote an answer line | see step 2's chart | **99.7 %** |
| Average answer length | — | 101 characters |

> **Teacher:** Almost every answer is now in the right format. But accuracy is 11 %. What does that tell us?
>
> **Student D:** It learned how to *answer*, but not how to *get it right*?
>
> **Teacher:** That's the lesson of the day. Format is cheap. Correctness is expensive, and for a model this small, most of it has to come from pretraining. SFT mostly *unlocks* what the base model already knows.
>
> **Student A:** Arithmetic went up, but path is still at zero.
>
> **Teacher:** The page gives you the hint: a family that stays at zero usually needs *more demonstrations of that family*, not more steps. Path needs the model to track a position over several moves. That's hard for a small model and needs a lot of examples.
>
> **Student B:** Why did it *lose* some questions it got right before?
>
> **Teacher:** Read the "Questions it lost" table. It's usually a family that's under-represented in the demonstrations, or the model's new habits (longer answers, the step format) sometimes lead it astray. Fine-tuning isn't only additive: it trades.

**Back to the sort prediction from step 1.** Compare the sort family between a with-steps and a without-steps run, and ask the student who predicted it to explain what they see.

**The leaderboard (screen 5)** ranks accuracy. Say explicitly that the project grader rewards *improvement over the base model*, not just the final number (§7), so a student with a weaker base isn't punished for it.

---

## 7. The project and how it's graded

Students can also hand in a whole training script (`project/train.py`). It must save `outputs/model.pt`. The grader:

- asks 300 unseen questions on a seed nobody sees;
- counts exact matches;
- compares against the base model recorded in the checkpoint.

| check | points |
|---|---|
| the checkpoint loads | 10 |
| accuracy (full marks at 80 %; the check passes at 30 %) | up to 60 |
| improvement over the base model (full marks at +40 points) | up to 30 |

The sample project (`experiments/world-sft/sample`) lists the levers, roughly in order: reasoning steps or not; the family mix (lookup is learned in a few hundred examples, path and shop take thousands); how long to train; and the prompt format, kept identical between training and evaluation.

---

## 8. Frequently asked questions

**"What's the difference between pretraining and fine-tuning, really? Isn't it the same training loop?"**
The loop is almost the same. The data and the goal differ. Pretraining sees raw text and learns language. Fine-tuning sees question-answer pairs, with the question masked, and learns a behaviour. It's also far shorter and uses a far smaller learning rate.

**"Why not just use a bigger model?"**
Try it: the prepared models go up to Qwen2.5-1.5B. A bigger base usually knows more, so SFT unlocks more. But it's slower, and on this server's CPU it's impractical. The point of the small model is that you can *see* every effect clearly and quickly.

**"What's LoRA, and why does the HuggingFace model use it?"**
Instead of changing all the weights, LoRA trains two small matrices beside each large one and adds their product in. Here that's rank 16 on the attention and MLP projections. It needs far less memory and produces a tiny file (the adapter). Experiment 7 compares it against full fine-tuning directly.

**"Is 11 % bad?"**
Compared with what? It's more than double the 5 % baseline, from a model the class trained from scratch in one session. The fair questions are how much it *gained* and *where*: the family chart answers both.

**"Could we just train on the test questions?"**
You'd get a high number that means nothing. The model would have memorised answers, not learned a task. That's why the held-out set and the project grader's secret seed exist.

**"Why does temperature 0 matter for evaluation?"**
At temperature 0 the model always picks its most likely token, so the same model gives the same answer every time. With sampling, two runs of the same evaluation would disagree, and a small difference between students could be pure luck.

**"Why does the validation loss go down but accuracy barely moves?"**
The loss is averaged over *every* target token: all the "First… Then… The answer is" text the model gets right easily. Accuracy depends on a handful of number tokens it often still gets wrong. Loss is a guide for *when to stop*. Accuracy is the result.

**"Can this work in Japanese?"**
Yes. The world generates Japanese questions and answers when the base model's language is Japanese, which comes from its tokenizer run or its meta.json. The grading works the same way.

**"My run was killed / timed out. What do I change?"**
Put your live view on the wall and read the reason together. For memory: a smaller batch, or keep the platform's training function, which already splits batches on a CPU. For time: fewer epochs. The progress line shows how long is left, so you can tell early.

---

## 9. Notes for the teacher: known quirks

These are properties of the experiment as currently built. They're worth knowing so a student's sharp question doesn't catch you out, and several make good discussion material.

1. **The sort family's reasoning steps are malformed** (§6, step 1). Treat it as a lesson about demonstration quality, or tell students to leave sort out when comparing with and without steps.
2. **On the HuggingFace (LoRA) path, the question isn't masked out of the loss.** The step trains through TRL's `SFTTrainer` with chat-formatted rows. TRL computes loss on the whole conversation unless `assistant_only_loss` is on, and it defaults to off. Masking as described in §2 applies to the class's own mini models. Fine for a lesson, but say so if a student asks.
3. **"Keep the instruction in the prompt" = off isn't carried into evaluation.** Step 4 falls back to the instruction from step 1's metadata when training didn't use one. So a no-instruction model is evaluated *with* the instruction, although the help text says evaluation follows the training choice.
4. **Free-text prepared datasets (Alpaca, Dolly) score close to 0 % by exact match**, whatever the model learns. Use GSM8K or the generated world when accuracy is the point.
5. **The sample project's validation targets omit the reasoning steps** ("Answer: x" only), while its training targets include them. Its printed validation loss is on a different kind of target from its training loss.
6. **The sample project's checkpoint doesn't record the base model**, so the grader scores its "improvement" on absolute accuracy instead of the real gain. Students who adapt it can add the base model's path to the checkpoint to be graded on true improvement.

---

## 10. Exit ticket

Ask each student to write one line for each:

1. One thing fine-tuning changed about the model, with the number that shows it.
2. One thing it *didn't* change, and what you'd try next (more demonstrations of a family? reasoning steps? fewer epochs?).
3. Why we ran the baseline before training.

### Homework (pick one)

- Run step 1 twice, with and without reasoning steps, keeping everything else the same, and compare step 4's family chart.
- Leave out the family your model did best on, and see whether the others improve. (Did the time spent on it matter?)
- Re-implement step 3 to stop at the best validation loss, and hand it in. Your result appears on the wall beside the standard code's.
