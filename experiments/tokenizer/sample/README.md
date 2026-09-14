# Build your own tokenizer

Your task: implement `Tokenizer` in `project/tokenizer.py` so that it

1. trains on a list of texts to a target vocabulary size,
2. round-trips exactly — `decode(encode(x)) == x` for any text,
3. compresses well — more characters per token than the reference BPE gets at the same vocabulary size,
4. encodes fast — the grader times 10,000 held-out documents.

The starter is a *character-level* tokenizer. It satisfies 1 and 2 and scores
badly on 3. Replace it with BPE, WordPiece, Unigram, or something of your own.

## Run and test
- `python project/train.py --vocab-size 2000` trains on the shipped pools and prints statistics (use the Run button).
- `python tests/quick_check.py` runs the interface checks locally (the Test button runs the full grader on the server).
- Submit when the grader is green. Every submission is graded automatically; the teacher can re-run it.

## Data
`atelier_sdk.material("datasets/tokenizer/...")` gives you the server-side paths of the prepared materials.
Put your own text under `data/` — it is included when you submit.
