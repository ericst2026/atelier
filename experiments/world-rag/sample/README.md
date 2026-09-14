# Beat keyword search on a corpus built to defeat it

`project/retriever.py` defines `build(documents) -> index` and
`search(index, queries, k) -> list[list[int]]` returning document indices.

The grader builds a library it has never shown you, scores recall@5 and recall@1,
and times indexing and search. Anything is allowed — BM25, your trained embedder,
both fused.

Where the marks are: the queries paraphrase their documents, so exact word
matching finds the cluster but not the member. Contrastive training on generated
pairs teaches the paraphrase mapping; mining hard negatives from BM25's mistakes
teaches the cluster distinction. Fusing the two is usually worth another few
points, because the embedder is weak on exactly the queries keyword search is
strong on — the ones turning on a rare literal string like a number.
