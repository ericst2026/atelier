# Tokenizer

Four guided steps (Corpus → Dedup → Train → Coverage) reproduce a real data
pipeline on CPU. Then build your own tokenizer in **My project** — the sample
project ships a character-level baseline that passes the interface check but
compresses poorly; your job is to beat the reference BPE on chars-per-token
while keeping exact round trips.

Reading list: Sennrich et al. 2016 (BPE for NMT), the GPT-2 byte-level
pre-tokenizer, Broder 1997 (MinHash), Lee et al. 2022 (deduplicating training
data makes LMs better).
