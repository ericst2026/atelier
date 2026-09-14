# Shape a reward, train a policy

`project/reward.py` — `reward(question, completion, gold) -> float`. The
starter gives 1.0 for a correct final number and 0.2 for marking one. Try
partial credit for correct intermediate numbers, penalties for length or
repetition, a bonus for a short clean final line … and watch what the policy
does with it.

`project/train.py` — GRPO from `Qwen2.5-0.5B-Instruct` on GSM8K prompts using
your reward, saving to `outputs/policy/`.

The grader (1) checks your reward prefers reference solutions over corrupted
ones on 60 problems and (2) measures your policy's accuracy on hidden GSM8K
problems. It also counts flagged completions.
