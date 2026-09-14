# Turn your base model into one that answers

`project/train.py` fine-tunes a base model and saves `outputs/model.pt`. The
grader asks it 300 generated questions it has never seen, counts exact matches,
and compares against the base model you started from — so the score is what *you*
added, not what pretraining already gave you.

Levers that matter, roughly in order:

- **reasoning steps in the target or not.** Steps cost tokens and time, and usually
  buy a lot of accuracy on the multi-step families.
- **the mix of families.** Equal shares are not optimal: `lookup` is learned in a
  few hundred examples, `path` and `shop` take thousands.
- **how long.** Small models overfit demonstrations quickly; watch the validation
  loss rather than the epoch count.
- **the prompt format**, kept identical between training and evaluation.

`python project/train.py --base <model.pt> --tokenizer <tokenizer.json>` runs the
whole thing and prints accuracy before and after.
