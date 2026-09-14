# Make the node earn its eight GPUs

`project/train_ddp.py` trains the reference model on however many GPUs it is given
(`ATELIER_GPUS`), saves `outputs/model.pt`, and writes `outputs/throughput.json`
with `{"tokens_per_sec", "gpus", "config"}`.

The grader runs it on one GPU and on four and scores throughput per GPU *and* the
loss reached. A recipe that is fast because it trains a worse model scores badly.

Where the time actually goes, in the order worth attacking: batch size (a GPU that
is not full is a GPU that is idle), the data path (if the loader is on the critical
path, nothing else matters), the all-reduce (accumulate gradients to do it less
often), activation memory (checkpoint, then spend the saving on a bigger batch),
and precision. `torch.compile` is worth a try and sometimes worth nothing.
