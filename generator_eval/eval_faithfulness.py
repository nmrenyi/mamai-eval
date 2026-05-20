"""Generator faithfulness evaluation — placeholder.

Measures whether the model's output is grounded in the retrieved RAG
context (i.e. claim-level support rate, hallucination rate). Distinct
from end-to-end generation quality, which compares the full output
against a gold reference.

Planned approach:
  - Run Gemma 4 E4B on questions with retrieved context (shared/inference.py
    + shared/prompts.py.build_rag_open_messages).
  - Decompose the model output into individual claims.
  - For each claim, ask a judge: "is this claim supported by the retrieved
    context?" (Yes / Partially / No).
  - Aggregate: per-row support rate, dataset-level hallucination rate.

See docs/refactor-plan-tracks.md for the broader plan.
"""


def main():
    raise NotImplementedError(
        "generator_eval/eval_faithfulness.py is a placeholder. "
        "See docs/refactor-plan-tracks.md for the planned scope."
    )


if __name__ == "__main__":
    main()
