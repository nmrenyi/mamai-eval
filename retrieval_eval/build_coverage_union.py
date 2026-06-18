"""Unify the 7 retrievers' kenya top-20 into per-retriever retrievals files (screen_embedder
coverage format), keyed by a normalized chunk-body hash so cross-retriever dedup works.

Sources (different formats):
  - pool files (mamaretrieval pool_candidates): candidates[].{retriever}_rank  → chunk_id
  - results files (lateon/voyage):             results[].{chunk_id,rank}
  - screen files (gecko/EmbeddingGemma, mine): datasets.<ds>[].chunks[].text   (sqlite body)

Chunk identity = sha256(normalized body)[:16]; canonical text = corpus body (chunk_id→body).
gecko/EG bodies map to the same key by stripping their [SOURCE|PAGE] prefix + whitespace-normalizing.
"""
import argparse, json, re, hashlib
from pathlib import Path

HEADER = re.compile(r"^\s*(?:<sep>)?\s*\[SOURCE:[^|\]]*\|PAGE:[^|\]]*(?:\|CID:[^\]]*)?\]", re.I)


def norm_body(text: str) -> str:
    t = re.sub(r"^\s*<sep>\s*", "", text)
    t = HEADER.sub("", t, count=1)
    return re.sub(r"\s+", " ", t).strip().lower()


def key_of(body: str) -> str:
    return hashlib.sha256(norm_body(body).encode("utf-8")).hexdigest()[:16]


def parse_corpus(path):
    raw = Path(path).read_text()
    cid2body, key2canon = {}, {}
    for seg in raw.split("<sep>"):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"\[SOURCE:([^|]+)\|PAGE:([^|]+)\|CID:([^\]]+)\]", seg)
        if not m:
            continue
        cid = m.group(3)
        body = seg[m.end():].strip()
        cid2body[cid] = body
        key2canon.setdefault(key_of(body), body)
    return cid2body, key2canon


def load_questions(qfile):
    out = {}
    with open(qfile) as f:
        for line in f:
            r = json.loads(line)
            out[r["query_id"]] = r["query_text"]
    return out


def add(out, qid, q, key, text, dataset="kenya"):
    rec = out.setdefault(qid, {"id": qid, "question": q, "chunks": [], "_seen": set()})
    if key not in rec["_seen"]:
        rec["_seen"].add(key)
        rec["chunks"].append({"idx": key, "text": text})


def finalize(out, name):
    recs = []
    for r in out.values():
        r.pop("_seen", None)
        recs.append(r)
    return {"candidate": name, "datasets": {"kenya": recs}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--questions", required=True, help="kenya_queries.jsonl")
    ap.add_argument("--pool", action="append", default=[], help="name=path for pool files (one file may hold multiple retrievers)")
    ap.add_argument("--results", action="append", default=[], help="name=path for results-format files")
    ap.add_argument("--screen", action="append", default=[], help="name=path for screen-format files (gecko/EG)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cid2body, key2canon = parse_corpus(args.corpus)
    print(f"corpus: {len(cid2body)} chunks", flush=True)
    qtext = load_questions(args.questions)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # pool files: discover retriever names from *_rank keys
    pool_out = {}  # retriever -> {qid: rec}
    for spec in args.pool:
        _, path = spec.split("=", 1)
        for line in open(path):
            d = json.loads(line)
            qid = d["query_id"]; q = qtext.get(qid, d.get("query_text", ""))
            rank_keys = {k[:-5] for c in d["candidates"] for k in c if k.endswith("_rank")}
            for rk in rank_keys:
                for c in d["candidates"]:
                    r = c.get(f"{rk}_rank")
                    if r is not None and r <= args.top_k:
                        body = cid2body.get(c["chunk_id"])
                        if body is None:
                            continue
                        add(pool_out.setdefault(rk, {}), qid, q, key_of(body), body)
    for rk, out in pool_out.items():
        p = outdir / f"{rk}.json"; p.write_text(json.dumps(finalize(out, rk))); written.append(rk)

    # results files (lateon/voyage)
    for spec in args.results:
        name, path = spec.split("=", 1)
        out = {}
        for line in open(path):
            d = json.loads(line)
            qid = d["query_id"]; q = qtext.get(qid, "")
            res = d["results"]
            if isinstance(res, str):  # may be JSON or a Python-literal (single-quoted) string
                try:
                    res = json.loads(res)
                except Exception:
                    import ast
                    res = ast.literal_eval(res)
            for c in res:
                if c["rank"] <= args.top_k:
                    body = cid2body.get(c["chunk_id"])
                    if body is None:
                        continue
                    add(out, qid, q, key_of(body), body)
        (outdir / f"{name}.json").write_text(json.dumps(finalize(out, name))); written.append(name)

    # screen files (gecko/EG): bodies are sqlite text with [SOURCE|PAGE] prefix
    miss = 0; tot = 0
    for spec in args.screen:
        name, path = spec.split("=", 1)
        d0 = json.loads(Path(path).read_text())
        out = {}
        for rec in d0["datasets"]["kenya"]:
            qid = rec["id"]; q = rec.get("question", qtext.get(qid, ""))
            for c in rec["chunks"][:args.top_k]:
                k = key_of(c["text"]); tot += 1
                canon = key2canon.get(k)
                if canon is None:
                    miss += 1
                    canon = c["text"]
                add(out, qid, q, k, canon)
        (outdir / f"{name}.json").write_text(json.dumps(finalize(out, name))); written.append(name)
    if tot:
        print(f"screen body→corpus alignment: {tot-miss}/{tot} matched ({100*(tot-miss)/tot:.1f}%)", flush=True)
    print("wrote retrievers:", written, "->", outdir, flush=True)


if __name__ == "__main__":
    main()
