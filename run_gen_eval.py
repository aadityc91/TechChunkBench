"""End-to-end generation evaluation on the technical regime (R2), mirroring the
conference paper's Section 3.4 (200 queries, top-5, BGE-large, 512 tokens).

Phase "retrieve": sample 200 QA pairs (stratified over the 24 corpora, seed 42,
    evidence locatable), retrieve top-5 chunks per strategy, record span-label
    MRR per query -> results/gen_eval/retrieval.jsonl
Phase "generate": generator = claude-haiku-4-5 (cheap, analogous to the
    conference's GPT-4o-mini); judge = claude-opus-5 scoring correctness (vs gold
    answer) and faithfulness (vs retrieved context) on 1-5 scales with structured
    output -> results/gen_eval/judged.jsonl (resumable, one line per (qa, strategy))
Phase "analyze": per-strategy means, Kruskal-Wallis, MRR-correctness correlation
    -> results/aggregated/gen_eval_summary.csv

Usage: python run_gen_eval.py retrieve [--strategies a,b] | generate | analyze
"""
import os, sys, json, random, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

from config import RAW_RESULTS_DIR, AGGREGATED_DIR
GEN_DIR = os.path.join(os.path.dirname(RAW_RESULTS_DIR), "gen_eval"); os.makedirs(GEN_DIR, exist_ok=True)
RET = os.path.join(GEN_DIR, "retrieval.jsonl"); JUD = os.path.join(GEN_DIR, "judged.jsonl")
MODEL_EMB = "BAAI/bge-large-en-v1.5"; SIZE = 512; K = 5; N_QUERIES = 200
LIMIT = 0
GENERATOR = "claude-haiku-4-5"; JUDGE = "claude-opus-5"
DEFAULT_STRATS = ["fixed_size", "fixed_overlap", "structure_aware", "semantic", "structure_noprefix"]

def load_env():
    p = os.path.expanduser("~/chunk-ext/.env")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)

# ---------------- phase 1 ----------------
def phase_retrieve(strategies):
    from src.chunkers import get_chunker
    from src.document_loader import load_all_corpora
    from src.embedder import embed_chunks, embed_queries, get_embed_fn
    from src.retriever import build_faiss_index, retrieve
    from src.qa_generator import load_qa_pairs
    from src import span_eval
    corpora = load_all_corpora(); rng = random.Random(42)
    # sample: round-robin over corpora until 200
    per = {}
    for cid, text in sorted(corpora.items()):
        doc_norm = span_eval.normalize(text); qa_all = load_qa_pairs(cid)["qa_pairs"]
        ok = [(qa, span_eval.locate_evidence(doc_norm, qa.get("evidence", ""))) for qa in qa_all]
        ok = [(qa, sp) for qa, sp in ok if sp is not None]; rng.shuffle(ok); per[cid] = ok
    sample = []; i = 0
    while len(sample) < N_QUERIES:
        for cid in sorted(per):
            if i < len(per[cid]) and len(sample) < N_QUERIES: sample.append((cid,) + per[cid][i])
        i += 1
    print(f"sampled {len(sample)} queries over {len(per)} corpora")
    done = set()
    if os.path.exists(RET):
        for line in open(RET): r = json.loads(line); done.add((r["qa_id"], r["strategy"]))
    out = open(RET, "a")
    for strategy in strategies:
        for cid, text in sorted(corpora.items()):
            items = [(qa, sp) for c, qa, sp in sample if c == cid and (qa["qa_id"], strategy) not in done]
            if not items: continue
            doc_norm = span_eval.normalize(text)
            chunker = get_chunker(strategy, target_size=SIZE, embed_fn=get_embed_fn(MODEL_EMB)) if strategy == "semantic" else get_chunker(strategy, target_size=SIZE)
            chunks = chunker.chunk(text, cid); texts = [c.text for c in chunks]
            spans = span_eval.locate_chunks(doc_norm, texts, [c.metadata.get("heading_path", "") for c in chunks])
            embs = embed_chunks(texts, MODEL_EMB, corpus_id=cid, strategy=f"e1_{strategy}", size=SIZE)
            index = build_faiss_index(embs); qe = embed_queries([qa["question"] for qa, _ in items], MODEL_EMB)
            for (qa, ev_span), q in zip(items, qe):
                idxs = [j for j in retrieve(q, index, top_k=K) if 0 <= j < len(texts)]
                labels = span_eval.relevance_labels([spans[j] for j in idxs], ev_span)
                out.write(json.dumps(dict(qa_id=qa["qa_id"], corpus_id=cid, strategy=strategy, question=qa["question"], gold_answer=qa["answer"],
                                          evidence=qa["evidence"], contexts=[texts[j] for j in idxs], labels=labels,
                                          mrr=span_eval.span_mrr(labels), hit5=span_eval.span_hit(labels, K))) + "\n"); out.flush()
            print(f"  {strategy:18s} {cid:16s} {len(items)} queries done")
    out.close()

# ---------------- phase 2 ----------------
GEN_SYS = ("You answer questions using only the provided context passages. Give a direct, concise answer (one to three sentences). "
           "If the context does not contain the answer, say so explicitly.")
def phase_generate():
    import anthropic
    from pydantic import BaseModel, Field
    class Judgment(BaseModel):
        correctness: int = Field(description="1-5: agreement of the generated answer with the gold answer (5 = fully correct and complete)")
        faithfulness: int = Field(description="1-5: degree to which every claim in the generated answer is supported by the retrieved context (5 = fully grounded)")
        rationale: str = Field(description="one or two sentences")
    load_env(); client = anthropic.Anthropic(max_retries=5)
    rows = [json.loads(l) for l in open(RET)]
    done = set()
    if os.path.exists(JUD):
        for l in open(JUD): r = json.loads(l); done.add((r["qa_id"], r["strategy"]))
    todo = [r for r in rows if (r["qa_id"], r["strategy"]) not in done]
    if LIMIT: todo = todo[:LIMIT]
    print(f"{len(rows)} retrieval rows, {len(todo)} to generate+judge")
    def work(r):
        ctx = "\n\n".join(f"[Passage {i+1}]\n{c}" for i, c in enumerate(r["contexts"]))
        g = client.messages.create(model=GENERATOR, max_tokens=300, system=GEN_SYS,
                                   messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {r['question']}"}])
        answer = "".join(b.text for b in g.content if b.type == "text").strip()
        jprompt = (f"Question: {r['question']}\n\nGold answer: {r['gold_answer']}\n\nRetrieved context:\n{ctx}\n\n"
                   f"Generated answer: {answer}\n\nScore the generated answer.")
        j = client.messages.parse(model=JUDGE, max_tokens=2000, output_config={"effort": "medium"}, output_format=Judgment,
                                  system="You are a strict, consistent grader of question-answering systems.",
                                  messages=[{"role": "user", "content": jprompt}])
        if j.stop_reason == "refusal" or j.parsed_output is None:
            return dict(qa_id=r["qa_id"], strategy=r["strategy"], corpus_id=r["corpus_id"], answer=answer, correctness=None, faithfulness=None, rationale="judge refused/unparsed", mrr=r["mrr"], hit5=r["hit5"])
        p = j.parsed_output
        return dict(qa_id=r["qa_id"], strategy=r["strategy"], corpus_id=r["corpus_id"], answer=answer, correctness=p.correctness, faithfulness=p.faithfulness,
                    rationale=p.rationale, mrr=r["mrr"], hit5=r["hit5"], gen_in=g.usage.input_tokens, gen_out=g.usage.output_tokens, judge_in=j.usage.input_tokens, judge_out=j.usage.output_tokens)
    out = open(JUD, "a"); n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for f in as_completed(futs):
            try: res = f.result()
            except Exception as e:
                r = futs[f]; print("ERROR", r["qa_id"], r["strategy"], repr(e)[:200]); continue
            out.write(json.dumps(res) + "\n"); out.flush(); n += 1
            if n % 50 == 0: print(f"  {n}/{len(todo)} judged, {time.time()-t0:.0f}s")
    out.close(); print("generate+judge done")

# ---------------- phase 3 ----------------
def phase_analyze():
    from scipy import stats
    df = pd.DataFrame([json.loads(l) for l in open(JUD)]).dropna(subset=["correctness"])
    g = df.groupby("strategy").agg(n=("qa_id", "count"), mrr=("mrr", "mean"), hit5=("hit5", "mean"), correctness=("correctness", "mean"), faithfulness=("faithfulness", "mean")).reset_index()
    core = [s for s in ["fixed_size", "fixed_overlap", "structure_aware", "semantic"] if s in set(df.strategy)]
    kw_c = stats.kruskal(*[df[df.strategy == s]["correctness"] for s in core]); kw_f = stats.kruskal(*[df[df.strategy == s]["faithfulness"] for s in core])
    r = stats.pearsonr(df["mrr"], df["correctness"]); rs = stats.spearmanr(df["mrr"], df["correctness"])
    # paired per query: structure_aware vs fixed_size correctness
    piv = df.pivot_table(index="qa_id", columns="strategy", values="correctness")
    extra = {}
    if {"structure_aware", "fixed_size"} <= set(piv.columns):
        d = (piv["structure_aware"] - piv["fixed_size"]).dropna(); extra = dict(sa_minus_fixed_correctness=d.mean(), wilcoxon_p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0)
    summ = dict(kruskal_correctness_H=kw_c.statistic, kruskal_correctness_p=kw_c.pvalue, kruskal_faithfulness_H=kw_f.statistic, kruskal_faithfulness_p=kw_f.pvalue,
                pearson_mrr_correctness=r[0], pearson_p=r[1], spearman_mrr_correctness=rs[0], n_rows=len(df), **extra)
    g.to_csv(f"{AGGREGATED_DIR}/gen_eval_by_strategy.csv", index=False); pd.DataFrame([summ]).to_csv(f"{AGGREGATED_DIR}/gen_eval_summary.csv", index=False)
    print(g.round(3).to_string(index=False)); print(pd.Series(summ).round(4).to_string())
    cost = df[["gen_in", "gen_out", "judge_in", "judge_out"]].sum() if "gen_in" in df else None
    if cost is not None: print("tokens:", cost.to_dict(), "approx USD:", round(cost.gen_in/1e6*1 + cost.gen_out/1e6*5 + cost.judge_in/1e6*5 + cost.judge_out/1e6*25, 2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["retrieve", "generate", "analyze"]); ap.add_argument("--strategies", default=",".join(DEFAULT_STRATS)); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); LIMIT = a.limit
    {"retrieve": lambda: phase_retrieve(a.strategies.split(",")), "generate": phase_generate, "analyze": phase_analyze}[a.phase]()
