# Teaching guides

Teachers' companions to the experiments. Each guide covers:

- the lesson's goal and a 90-minute plan;
- what to put on the wall at each point;
- sample classroom conversations with the questions students usually ask;
- run times on a GPU and on a CPU-only server;
- known quirks of the experiment as it is built.

| guide | experiment | builds on |
|---|---|---|
| [Pretraining](pretraining.md) | `experiments/world-pretrain`: train a small transformer from scratch | a Tokenizer run |
| [Fine-tuning](finetuning.md) | `experiments/world-sft`: teach the base model to answer | a Pretraining run, or a prepared model |

Taught in that order, the class's own pretrained model becomes the base model it fine-tunes.
