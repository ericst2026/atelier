# Make Qwen2.5-0.5B an assistant

Produce `outputs/adapter/` (LoRA) or `outputs/model/` (full checkpoint) with
`project/train.py`. The grader scores held-out loss on a hidden validation set
and a judge's preference against the untuned base model on 30 prompts.

Ideas that move the score: pick and clean data (quality beats quantity at this
scale), write a system prompt and keep it at inference, mask the prompt tokens
from the loss, tune the learning rate, train on multi-turn data, curriculum by
length.

`data/` in your project is included in the submission; put JSONL there.
