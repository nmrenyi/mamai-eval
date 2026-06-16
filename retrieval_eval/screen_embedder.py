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


def _retrieve_loop(args, texts, doc_emb, enc_queries, out, mrl):
    for dataset in [d.strip() for d in args.datasets.split(",")]:
        queries = load_queries(args.hf_repo, args.revision, dataset)
        print(f"[{dataset}] {len(queries)} queries", flush=True)
        q_emb = enc_queries([q for _, q in queries])
        q_emb = _mrl(q_emb, args.dim) if mrl else np.asarray(q_emb, dtype=np.float32)
        sims = q_emb @ doc_emb.T
        topk = np.argsort(-sims, axis=1)[:, :args.top_k]
        out["datasets"][dataset] = [
            {"id": qid, "question": qt,
             "chunks": [{"idx": int(j), "text": texts[j], "sim": float(sims[i, j])} for j in topk[i]]}
            for i, (qid, qt) in enumerate(queries)]


def embed_retrieve(args):
    # Gecko baseline arm: reuse the app's STORED doc vectors (true deployed behavior) +
    # the Gecko TFLite for query encoding. Directly comparable to ST candidates below.
    if args.candidate == "gecko":
        from retrieval_eval.retrieval import load_vector_store, GeckoEmbedder
        store = load_vector_store(args.db_path)
        texts = [t for t, _ in store]
        doc_emb = np.asarray([e for _, e in store], dtype=np.float32)
        doc_emb /= (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
        emb = GeckoEmbedder(args.gecko_model, args.tokenizer)
        def enc_queries(xs):
            q = np.asarray([emb.embed(x) for x in xs], dtype=np.float32)
            return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        out = {"candidate": "gecko", "dim": int(doc_emb.shape[1]), "datasets": {}}
        _retrieve_loop(args, texts, doc_emb, enc_queries, out, mrl=False)
        Path(args.out).write_text(json.dumps(out)); print("wrote", args.out, flush=True); return

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
    _retrieve_loop(args, texts, doc_emb, enc_queries, out, mrl=True)
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


def arm_format(args):
    """Convert retrievals.json -> value-gate arm dir (top-k formatted chunk strings)."""
    from retrieval_eval.retrieval import format_app_context_chunks
    data = json.loads(Path(args.retrievals).read_text())
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    arm = data.get("candidate", "candidate")
    for ds, recs in data["datasets"].items():
        retr = []
        for rec in recs:
            top = rec["chunks"][:args.top_k]
            cc, docs = format_app_context_chunks([c["text"] for c in top])
            retr.append({"id": rec["id"], "question": rec["question"], "chunks": cc,
                         "retrieved_docs": docs, "chunk_indices": [c["idx"] for c in top]})
        (outdir / f"{ds}.json").write_text(json.dumps(
            {"metadata": {"dataset": ds, "arm": arm},
             "config": {"top_k": args.top_k, "n_questions": len(retr), "arm": arm},
             "retrievals": retr}, ensure_ascii=False))
    (outdir / "manifest.json").write_text(json.dumps(
        {"schema_version": 2, "arm": arm,
         "datasets": {ds: {"output_file": f"{ds}.json", "n_questions": len(recs)}
                      for ds, recs in data["datasets"].items()}}) + "\n")
    print("wrote arm to", outdir, flush=True)


def coverage(args):
    """Top-k union coverage across multiple retrievals files (corpus-absent vs buried split)."""
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    files = [f.strip() for f in args.retrievals.split(",")]
    arms, qtext = {}, {}
    for f in files:
        d = json.loads(Path(f).read_text()); name = d.get("candidate", f)
        for ds, recs in d["datasets"].items():
            for rec in recs:
                qtext[rec["id"]] = rec["question"]
                arms.setdefault(name, {}).setdefault(rec["id"], []).extend(
                    (c["idx"], c["text"]) for c in rec["chunks"][:args.top_k])
    uniq = {}
    for name, qmap in arms.items():
        for qid, chunks in qmap.items():
            for idx, text in chunks:
                uniq.setdefault((qid, idx), (qtext[qid], text))
    print(f"{len(qtext)} queries, {len(arms)} arms, {len(uniq)} unique (q,chunk) pairs to judge", flush=True)

    def judge(qt, text):
        r = client.chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=2048,
            messages=[{"role": "system", "content": V2_SYSTEM_PROMPT},
                      {"role": "user", "content": _build_user_content(qt, {"text": text})}])
        return parse_judge(r.choices[0].message.content)

    # resumable checkpoint (survives pod preemption/restart): "qid\tidx" -> grade
    ckpt = Path(args.out + ".ckpt.json")
    grades = {}
    if ckpt.exists():
        try:
            for k, v in json.loads(ckpt.read_text()).items():
                qid, idx = k.split("\t", 1)
                grades[(qid, int(idx) if idx.lstrip("-").isdigit() else idx)] = v
            print(f"resumed {len(grades)} grades from checkpoint", flush=True)
        except Exception as e:
            print("ckpt load failed, starting fresh:", e, flush=True)

    def save_ckpt():
        tmp = Path(args.out + ".ckpt.tmp")
        tmp.write_text(json.dumps({f"{q}\t{i}": g for (q, i), g in grades.items()}))
        tmp.replace(ckpt)

    items = [(k, v) for k, v in uniq.items() if k not in grades]
    print(f"  {len(items)} to judge ({len(grades)} already cached)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge, qt, tx): k for k, (qt, tx) in items}
        done = 0
        for fut in as_completed(futs):
            try: grades[futs[fut]] = fut.result()
            except Exception: grades[futs[fut]] = None
            done += 1
            if done % 500 == 0:
                save_ckpt(); print(f"  judged {done}/{len(items)}", flush=True)
    save_ckpt()

    qids = sorted(qtext)
    arm_names = list(arms)
    def covered(qid, names, thr):
        for nm in names:
            for idx, _ in arms[nm].get(qid, []):
                g = grades.get((qid, idx))
                if g is not None and g >= thr: return True
        return False
    rep = {"model": args.model, "top_k": args.top_k, "arms": arm_names, "n_queries": len(qids), "by_cut": {}}
    for cut, thr in [("lenient", 3), ("strict", 5)]:
        union = sum(covered(q, arm_names, thr) for q in qids)
        per_arm = {nm: sum(covered(q, [nm], thr) for q in qids) for nm in arm_names}
        # split vs deployed gecko if present
        base = "gecko" if "gecko" in arm_names else arm_names[0]
        buried = sum(1 for q in qids if covered(q, arm_names, thr) and not covered(q, [base], thr))
        absent = sum(1 for q in qids if not covered(q, arm_names, thr))
        rep["by_cut"][cut] = {
            "union_covered": round(union / len(qids), 4),
            "per_arm_covered": {nm: round(v / len(qids), 4) for nm, v in per_arm.items()},
            f"ranking_fixable_vs_{base}": round(buried / len(qids), 4),
            "corpus_absent": round(absent / len(qids), 4)}
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    cov = sub.add_parser("coverage")
    cov.add_argument("--retrievals", required=True, help="comma-separated retrievals files")
    cov.add_argument("--base-url", required=True)
    cov.add_argument("--model", default="Qwen/Qwen3-32B")
    cov.add_argument("--top-k", type=int, default=20)
    cov.add_argument("--workers", type=int, default=16)
    cov.add_argument("--out", required=True)
    cov.set_defaults(func=coverage)
    a = sub.add_parser("arm_format")
    a.add_argument("--retrievals", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--top-k", type=int, default=3)
    a.set_defaults(func=arm_format)
    e = sub.add_parser("embed_retrieve")
    e.add_argument("--candidate", required=True, help="HF model id, or 'gecko' for the deployed baseline")
    e.add_argument("--db-path", required=True)
    e.add_argument("--gecko-model", default="", help="Gecko .tflite (only for --candidate gecko)")
    e.add_argument("--tokenizer", default="", help="sentencepiece model (only for --candidate gecko)")
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
