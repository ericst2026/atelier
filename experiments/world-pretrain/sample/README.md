# Train a language model from scratch

Budget: **30M parameters**, your own tokenizer, and whatever tokens you can push
through in your slot. The grader loads `outputs/model.pt`, checks the budget, and
measures validation loss on a token stream it packs itself with the reference
tokenizer and a seed you have not seen.

- `project/model.py` — start from the reference `MiniLM` and change it: depth
  against width, the MLP ratio, untied embeddings, no RoPE, a different
  normalisation. Every choice moves the loss, and at this size you can measure it
  in minutes rather than guess.
- `project/train.py` — the loop is yours: schedule, batch size, context length,
  how much of the budget goes to stories against worked problems.

Cheap wins, roughly in order: get the learning rate right, spend the parameter
budget on layers rather than on a large vocabulary, train on more tokens rather
than a bigger model, and keep the context short enough that batches stay large.
