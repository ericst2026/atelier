# Design a reward, then live with it

`project/reward.py` — `reward(task, completion) -> float`. `task` gives you
`prompt`, `answer`, `family` and `steps`. Only differences within a group matter,
so the scale is yours.

Things worth trying: partial credit when an intermediate number in the working is
right; a bonus for a short final line; a penalty that grows with length rather
than a flat one; different weights per family; a reward that pays for agreeing
with a second sample of itself.

Things that look clever and are not: paying for the answer marker without paying
more for correctness, and any bonus the model can collect without solving
anything. The grader tests exactly that — it checks your reward prefers a correct
answer, and that an empty or enormous completion cannot outscore a right one.

`python project/train.py --policy <model.pt> --tokenizer <tokenizer.json> --steps 300`
