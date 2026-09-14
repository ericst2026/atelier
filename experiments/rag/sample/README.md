# Find the passage, then read it honestly

`project/retriever.py` — `build(corpus) -> index` and `search(index, question, k) -> list[dict]`
with at least `{"text", "doc_id"}` per result.
`project/answer.py` — `answer(question, passages, generate) -> str`.

The grader scores recall@5 on unseen questions, exact match on the answers, and how
often your system invents an answer to a question the corpus cannot answer. All
three matter: a retriever that returns everything scores nothing, and a reader that
always answers scores badly on the third.

Things that move recall: chunk size and overlap, the query prefix the embedder was
trained with, a reranker over the top thirty, and hybrid retrieval — keyword
matching catches the questions that share vocabulary with the passage, embeddings
catch the ones that do not, and together they beat either.
