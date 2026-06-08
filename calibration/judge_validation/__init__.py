"""Judge validation: meta-eval the rubric LLM judge against physician ground truth.

See docs/judge-validation-plan.md for design. Atomic unit is one criterion ->
one binary verdict, matching end_to_end_eval/rescore_rubric.py.
"""
