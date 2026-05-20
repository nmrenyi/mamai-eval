"""Retrieval quality evaluation — placeholder.

Planned metrics (to be implemented):
  - precision@k, recall@k, MRR on top-k retrieved chunks vs gold-relevant
    chunks (when ground-truth-relevant chunks are available)
  - NDCG@k if scores are continuous

Consumes rag_contexts/<context_version>/<dataset>.json produced by
retrieval_eval/precompute_retrieval.py.

See docs/refactor-plan-tracks.md for the broader plan.
"""


def main():
    raise NotImplementedError(
        "retrieval_eval/eval_retrieval.py is a placeholder. "
        "See docs/refactor-plan-tracks.md for the planned scope."
    )


if __name__ == "__main__":
    main()
