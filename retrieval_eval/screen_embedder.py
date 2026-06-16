"""R2c Phase 2 — offline retrieval screen for a candidate on-device embedder.

Two modes (run sequentially so the embedder frees the GPU before vLLM judging):

  embed_retrieve : embed the corpus (embeddings.sqlite texts) + dataset queries with a
                   SentenceTransformer candidate, cosine top-k retrieve, write retrievals.json.
  judge_score    : judge each (query, chunk) with Qwen3-32B + the V2 rubric via a local vLLM
                   endpoint (same judge/rubric as the 230k audit + kenya_relevance), then score:
                     P@3 (lenient>=3, strict>=5) + mean grade over top-3   [vs Gecko baseline]
                     HR@k (any grade>=3 / >=5 in top-k)                     [recall proxy = the
                                                                            embedder's unique lever]

Gecko kenya baseline to beat (top-3, same judge): P@3 lenient 0.277 / strict 0.180 / mean 1.49.
"""
import argparse, json, re, sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from retrieval_eval._kenya_rubric import V2_SYSTEM_PROMPT, _build_user_content, v2_score


# ----- corpus + queries -----
def load_corpus_texts(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT text FROM rag_vector_store").fetchall()
    conn.close()
    return [r[0] for r in rows]


def load_queries(hf_repo, revision, dataset):
    from datasets import load_dataset
    ds = load_dataset(hf_repo, dataset, revision=revision)
    split = "test" if "test" in ds else list(ds.keys())[0]
    out = []
    for r in ds[split]:
        qid = r.get("id") or (r.get("source") or {}).get("id") or str(len(out))
        q = r.get("question") or ""
        if isinstance(q, list):  # multi-turn safety; take last user turn
            q = next((t.get("content", "") for t in reversed(q) if t.get("role") == "user"), "")
        if q:
            out.append((str(qid), q))
    return out


def _mrl(emb, dim):
    if not dim:
        return np.asarray(emb, dtype=np.float32)
    emb = np.asarray(emb, dtype=np.float32)[:, :dim]
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def embed_retrieve(args):
    from sentence_transformers import SentenceTransformer
    texts = load_corpus_texts(args.db_path)
    print(f"corpus: {len(texts)} chunks", flush=True)
    model = SentenceTransformer(args.candidate, trust_remote_code=True, device="cuda")

    def enc_docs(xs):
        if hasattr(model, "encode_document"):
            return model.encode_document(xs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return model.encode([args.doc_prefix + t for t in xs], batch_size=64,
                            show_progress_bar=True, normalize_embeddings=True)

    def enc_queries(xs):
        if hasattr(model, "encode_query"):
            return model.encode_query(xs, batch_size=64, normalize_embeddings=True)
        return model.encode([args.query_prefix + q for q in xs], batch_size=64, normalize_embeddings=True)

    doc_emb = _mrl(enc_docs(texts), args.dim)
    out = {"candidate": args.candidate, "dim": args.dim or int(doc_emb.shape[1]), "datasets": {}}
    for dataset in [d.strip() for d in args.datasets.split(",")]:
        queries = load_queries(args.hf_repo, args.revision, dataset)
        print(f"[{dataset}] {len(queries)} queries", flush=True)
        q_emb = _mrl(enc_queries([q for _, q in queries]), args.dim)
        sims = q_emb @ doc_emb.T
        topk = np.argsort(-sims, axis=1)[:, :args.top_k]
        recs = [{"id": qid, "question": qt,
                 "chunks": [{"idx": int(j), "text": texts[j], "sim": float(sims[i, j])} for j in topk[i]]}
                for i, (qid, qt) in enumerate(queries)]
        out["datasets"][dataset] = recs
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out, flush=True)


# ----- judge + score -----
def parse_judge(text):
    m = re.search(r"\{[^{}]*d1_topic[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    def i(v):
        if isinstance(v, bool):
            return int(v)
        try:
            return int(v)
        except Exception:
            return 0
    return v2_score(bool(d.get("d1_topic", False)), i(d.get("d2_meaningful", 0)),
                    i(d.get("d3_actionable", 0)), i(d.get("d4_density", 0)))


def judge_score(args):
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    data = json.loads(Path(args.retrievals).read_text())

    def judge(qt, text):
        r = client.chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=2048,
            messages=[{"role": "system", "content": V2_SYSTEM_PROMPT},
                      {"role": "user", "content": _build_user_content(qt, {"text": text})}])
        return parse_judge(r.choices[0].message.content)

    report = {"candidate": data["candidate"], "dim": data["dim"], "model": args.model, "datasets": {}}
    for dataset, recs in data["datasets"].items():
        tasks = [(rec["id"], c["idx"], rec["question"], c["text"]) for rec in recs for c in rec["chunks"]]
        grades = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge, qt, tx): (qid, idx) for qid, idx, qt, tx in tasks}
            done = 0
            for fut in as_completed(futs):
                try:
                    grades[futs[fut]] = fut.result()
                except Exception:
                    grades[futs[fut]] = None
                done += 1
                if done % 200 == 0:
                    print(f"[{dataset}] judged {done}/{len(tasks)}", flush=True)
        pl = ps = mg = nq = hr3 = hr5 = 0
        for rec in recs:
            gs = [grades.get((rec["id"], c["idx"])) for c in rec["chunks"]]
            gs = [g for g in gs if g is not None]
            if not gs:
                continue
            nq += 1
            top3 = gs[:3]
            pl += sum(g >= 3 for g in top3) / 3.0
            ps += sum(g >= 5 for g in top3) / 3.0
            mg += sum(top3) / len(top3)
            hr3 += 1 if any(g >= 3 for g in gs) else 0
            hr5 += 1 if any(g >= 5 for g in gs) else 0
        k = len(recs[0]["chunks"]) if recs else 0
        report["datasets"][dataset] = {
            "n": nq, "top_k": k,
            "p_at_3_lenient": round(pl / nq, 4), "p_at_3_strict": round(ps / nq, 4),
            "mean_grade_top3": round(mg / nq, 4),
            f"hr_at_{k}_lenient": round(hr3 / nq, 4), f"hr_at_{k}_strict": round(hr5 / nq, 4)}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("embed_retrieve")
    e.add_argument("--candidate", required=True)
    e.add_argument("--db-path", required=True)
    e.add_argument("--datasets", default="kenya")
    e.add_argument("--hf-repo", default="nmrenyi/mamabench")
    e.add_argument("--revision", default="v0.2")
    e.add_argument("--top-k", type=int, default=20)
    e.add_argument("--dim", type=int, default=0, help="MRL truncation dim (0 = native)")
    e.add_argument("--query-prefix", default="")
    e.add_argument("--doc-prefix", default="")
    e.add_argument("--out", required=True)
    e.set_defaults(func=embed_retrieve)
    j = sub.add_parser("judge_score")
    j.add_argument("--retrievals", required=True)
    j.add_argument("--base-url", required=True)
    j.add_argument("--model", default="Qwen/Qwen3-32B")
    j.add_argument("--workers", type=int, default=16)
    j.add_argument("--out", required=True)
    j.set_defaults(func=judge_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
