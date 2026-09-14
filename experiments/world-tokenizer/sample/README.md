# Build the tokenizer the rest of the course will use

Implement `Tokenizer` in `project/tokenizer.py`. The grader trains it on text
generated from the same world (with a seed you have not seen), then measures:

- **round trip** — every held-out document must decode back exactly
- **compression** — characters per token against the reference BPE at the same vocabulary size
- **digits** — numbers encoded consistently, because four of the five task families are arithmetic underneath
- **speed** — the encoder runs over the whole corpus in pretraining, twice

The starter is character-level: correct, and poor at everything else. Replace it.

`python project/train.py --vocab-size 4096 --lang en` trains on generated text and
prints the same statistics the grader uses. Try `--lang ja` too: without spaces,
anything that splits on whitespace has nothing to work with.
