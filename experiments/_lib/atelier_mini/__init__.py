"""A complete small-LM stack in pure PyTorch: no transformers, no trl, no
pretrained weights. Everything a 10M–100M model needs, written to be read.

    model.py   the decoder-only transformer and its size presets
    tok.py     wraps a BPE trained in the Tokenizer experiment
    data.py    packing text into a token stream and sampling batches
    gen.py     batched sampling with temperature, top-k and a stop string
    train.py   the pretraining loop, and SFT with the prompt masked out
    rl.py      GRPO: sample a group, score it, push the above-average ones up
"""
