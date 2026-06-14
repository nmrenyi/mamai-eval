#!/usr/bin/env python3
"""R2c diagnostic — judge the relevance of each arm's kenya top-3 chunks.

The value gate showed reranking does not improve kenya answers, but the offline
P@3 win was measured on mamaretrieval (a different query set). This judges the
kenya/afrimedqa_saq retrievals directly — with the SAME V2 rubric + judge model
(Qwen3-32B) that produced the 230k grades — to learn whether the rerankers
actually surface more relevant chunks on kenya:

  - rerankers show higher kenya P@3 but answers didn't move  -> Gemma 4 ceiling
  - rerankers show same/lower kenya P@3                      -> reranker doesn't
                                                               transfer to kenya

Judges the deduped union of all four arms' top-3 (query, chunk) pairs once, then
computes per-arm P@3 (lenient grade>=3, strict grade>=5) + mean grade.

Usage (against a served Qwen3-32B):
  python -m retrieval_eval.judge_kenya_relevance \\
      --arms-root /lightscratch/.../rag_arms \\
      --mxbai-arms /lightscratch/.../rag_arms_mxbai \\
      --datasets kenya,afrimedqa_saq \\
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B \\
      --out kenya_relevance.json
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from retrieval_eval._kenya_rubric import V2_SYSTEM_PROMPT, _build_user_content, v2_score

# arm key -> (root selector, arm subdir). mxbai arm lives under a different root.
ARMS = [
    ("gecko",       "main",  "gecko"),
    ("hybrid",      "main",  "hybrid"),
    ("minilm_ft",   "main",  "hybrid_rerank"),
    ("mxbai_ft",    "mxbai", "hybrid_rerank"),
]


def load_arm(root, arm_sub, ds):
    p = Path(root) / arm_sub / f"{ds}.json"
    rows = json.load(open(p))["retrievals"]
    # query_id -> list of (chunk_index, chunk_dict{source,page,text}) for its top-3
    out = {}
    for r in rows:
        docs = r.get("retrieved_docs", [])
        idxs = r.get("chunk_indices", list(range(len(docs))))
        out[r["id"]] = {"q": r["question"],
                        "chunks": [(idxs[k], docs[k]) for k in range(len(docs))]}
    return out


def parse_judge(text):
    """Extract the JSON dims from the model output (tolerant)."""
    m = re.search(r"\{[^{}]*d1_topic[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    def i(v):
        if isinstance(v, bool): return int(v)
        try: return int(v)
        except Exception: return 0
    return v2_score(bool(d.get("d1_topic", False)),
                    i(d.get("d2_meaningful", 0)), i(d.get("d3_actionable", 0)),
                    i(d.get("d4_density", 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms-root", required=True)
    ap.add_argument("--mxbai-arms", required=True)
    ap.add_argument("--datasets", default="kenya,afrimedqa_saq")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    def judge(query, chunk):
        r = client.chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=2048,
            messages=[{"role": "system", "content": V2_SYSTEM_PROMPT},
                      {"role": "user", "content": _build_user_content(query, chunk)}])
        return parse_judge(r.choices[0].message.content)

    report = {"model": args.model, "datasets": {}}
    for ds in [d.strip() for d in args.datasets.split(",")]:
        arms = {}
        for key, sel, sub in ARMS:
            arms[key] = load_arm(args.mxbai_arms if sel == "mxbai" else args.arms_root, sub, ds)
        qids = sorted(arms["gecko"].keys())

        # dedup (qid, chunk_index) -> (query, chunk_dict)
        uniq = {}
        for key, _, _ in ARMS:
            for qid in qids:
                entry = arms[key].get(qid)
                if not entry:
                    continue
                for cidx, cdoc in entry["chunks"]:
                    uniq.setdefault((qid, cidx), (entry["q"], cdoc))
        print(f"[{ds}] {len(qids)} queries, {len(uniq)} unique (query,chunk) pairs to judge", flush=True)

        grades = {}
        items = list(uniq.items())
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge, q, c): k for k, (q, c) in items}
            done = 0
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    grades[k] = fut.result()
                except Exception as e:
                    grades[k] = None
                done += 1
                if done % 200 == 0:
                    print(f"  [{ds}] judged {done}/{len(items)}", flush=True)
        graded = {k: v for k, v in grades.items() if v is not None}
        print(f"[{ds}] judged ok: {len(graded)}/{len(items)}", flush=True)

        # per-arm P@3 (lenient >=3, strict >=5) + mean grade, over the top-3
        ds_rec = {"n_queries": len(qids), "n_pairs_judged": len(graded), "by_arm": {}}
        for key, _, _ in ARMS:
            pl = ps = mg = nq = 0
            for qid in qids:
                entry = arms[key].get(qid)
                if not entry:
                    continue
                gs = [grades.get((qid, cidx)) for cidx, _ in entry["chunks"]]
                gs = [g for g in gs if g is not None]
                if not gs:
                    continue
                nq += 1
                pl += sum(g >= 3 for g in gs) / 3.0
                ps += sum(g >= 5 for g in gs) / 3.0
                mg += sum(gs) / len(gs)
            ds_rec["by_arm"][key] = {
                "p_at_3_lenient": round(pl / nq, 4), "p_at_3_strict": round(ps / nq, 4),
                "mean_grade": round(mg / nq, 4), "n": nq}
        report["datasets"][ds] = ds_rec
        for key, _, _ in ARMS:
            b = ds_rec["by_arm"][key]
            print(f"  [{ds}] {key:10} P@3(>=3)={b['p_at_3_lenient']} "
                  f"P@3(>=5)={b['p_at_3_strict']} mean={b['mean_grade']}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("Written:", args.out, flush=True)


if __name__ == "__main__":
    main()
