# Financial Research Tool using RAG

A retrieval-augmented generation pipeline over Arm Holdings plc's FY2025 Form
10-K. It answers factual questions about the filing, cites the printed page
behind every claim, and refuses when the filing does not contain the answer.

> My RAG app helps **investment analysts** answer **factual questions about
> results, risk factors and governance** from **Arm Holdings' 201-page FY2025
> Form 10-K** in a **command-line assistant**, with a **95% faithfulness**
> target and a **3-second p95 latency** ceiling.

Measured: **100% faithfulness**, **100% correct refusal**, **100% citation
validity**, **p50 ~1.8s / p95 ~2.3s**.

**[Read the evaluation report →](https://jaynahalai.github.io/Financial_Research_Tool_using_RAG/)**

Charts, the full 18-configuration table, and the failure analysis. Source:
[`evaluation/report.html`](evaluation/report.html).

---

## The framework

| Field | Decision |
|---|---|
| **Use case** | An analyst asks a factual question about Arm's FY2025 filing and needs an answer with a page reference they can check. Runs as a CLI assistant; the retrieval layer is importable for any other surface. |
| **Corpus** | One document: Arm Holdings plc Form 10-K for the fiscal year ended 31 March 2025, saved from the SEC filing page as HTML. 136,349 words across printed pages 5–201. English. Source of truth is the filed document itself. |
| **Ingestion + cleaning** | `inspect_document.py` parses the HTML with BeautifulSoup, decomposes the `ix:header` and `ix:hidden` XBRL blocks, strips site navigation ("Download PDF" and similar), and collapses blank lines into one clean text file. |
| **Ingestion + freshness** | A 10-K is an annual document, so the corpus is refreshed manually once a year when the new filing appears. There is no automated refresh and no freshness SLA; this is a deliberate match to how often the source actually changes. |
| **Chunking + embedding** | Two strategies, compared head to head. Fixed: 500-word windows, 50-word overlap, 303 chunks. Semantic: embedding-similarity breakpoints at the 80th percentile, grouped toward 500 words, 258 chunks. Both embedded with `text-embedding-3-small` (1,536-dim) — small enough to be cheap over a 136k-word corpus, large enough to carry a 500-word chunk. |
| **Retrieve** | FAISS `IndexFlatIP` over L2-normalised vectors (cosine) for dense, BM25Okapi for sparse, fused with reciprocal rank fusion (k=60). Top 20 candidates go to a cross-encoder reranker; top 5 reach the generator. |

---

## Architecture

```
SEC filing (HTML)
    │
    ├─ inspect_document.py ──────────► data/raw/arm_report.txt
    │                                   (cleaned, 136,349 words)
    │
    ├─ build_fixed_chunks.py ────────► data/fixed_chunks.json     (303)
    ├─ build_semantic_chunks.py ─────► data/semantic_chunks.json  (258)
    │      └─ page_map.py attaches printed page numbers to every chunk
    │
    ├─ build_vector_store.py ────────► data/index/*.faiss
    │
    └─ retrieval.py
           dense (FAISS) ─┐
                          ├─ reciprocal rank fusion ─► 20 candidates
           sparse (BM25) ─┘                                │
                                                           ▼
                                        cross-encoder over 250-word windows
                                                           │
                                                           ▼
                                                    top 5 chunks
                                                           │
                            rag_pipeline.py ───────────────┘
                              gpt-4o-mini, cites pages, refuses when unsupported
```

### Page citations

Financial answers are worthless without a source, so every chunk carries the
printed pages it came from. `page_map.py` recovers them without a PDF: the
cleaned text still contains the filing's page footers as standalone numbers,
mixed in with hundreds of table figures. The real footers are the ones that
ascend through the document, so the map is the longest strictly increasing
subsequence of those candidates — 197 markers covering pages 5 to 201. Every
gold span in the benchmark resolves to a page through this map.

---

## Setup

Requires Python 3.9+ and an OpenAI API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env
```

## Running it

The build steps are ordered; each depends on the one before.

```bash
python inspect_document.py         # HTML  -> cleaned text
python build_fixed_chunks.py       # text  -> fixed chunks
python build_semantic_chunks.py    # text  -> semantic chunks   (calls the API)
python build_vector_store.py       # chunks -> FAISS indexes    (calls the API)
```

Then evaluate, or ask it questions:

```bash
python evaluate_retrieval.py       # 18 configurations -> evaluation/retrieval_results.json
python evaluate_answers.py         # end-to-end quality -> evaluation/answer_results.json
python rag_pipeline.py             # interactive Q&A
```

Embeddings are computed once at build time and persisted, so a question costs
one embedding call rather than one per chunk.

Two guards protect the index from going quietly stale. If the chunk files are
rebuilt without rebuilding the index, `retrieval.py` refuses to start rather
than return misaligned passages. And `data/index/manifest.json` records the
model the index was built with, so querying it with a different embedding model
fails loudly instead of silently degrading retrieval.

### Reproducibility

The pipeline has been re-run end to end from the source HTML. The cleaned text
and both chunk files reproduce **byte-identically**. The FAISS indexes do not —
OpenAI embeddings vary in the last few floating-point places between calls —
but all 18 retrieval configurations and every answer-quality metric reproduce
exactly. Only latency moves, by roughly 0.2s run to run.

---

## Results

Ten answerable questions, scored by strict verbatim span containment: a
required span counts only if it appears word-for-word in the retrieved text.

### Chunking (dense retrieval, no reranking)

| Strategy | @3 | @5 | @10 |
|---|---:|---:|---:|
| Fixed 500-word windows | 0.50 | 0.60 | 0.75 |
| Semantic breakpoints | **0.80** | **0.80** | **0.80** |

### Reranking (semantic chunks, hybrid retrieval)

| Reranking | @3 | @5 | @10 |
|---|---:|---:|---:|
| None | 0.70 | 0.70 | 0.90 |
| Whole chunk (truncated) | 0.60 | 0.60 | 0.80 |
| 250-word windows | **0.90** | **0.90** | **1.00** |

### End-to-end

| Metric | Result |
|---|---|
| Faithfulness (judged by `gpt-4o`) | 100% — 10 of 10 |
| Correct refusal | 100% — 5 of 5 unanswerable |
| False refusal | 0% — 0 of 10 answerable |
| Citation validity | 100% |
| Latency | p50 ~1.8s, p95 ~2.3s (warm) |

Full results for all 18 configurations: `evaluation/retrieval_results.json`.

---

## What the evaluation actually taught me

**Reranking only helps if the reranker can read the chunk.** The cross-encoder
truncates at 512 tokens, but the largest semantic chunk is 2,798 tokens of
dense financial text — so scoring a whole chunk judged it on roughly its first
fifth. That made reranking look actively harmful (0.70 → 0.60). Scoring
overlapping 250-word windows and taking each chunk's best window turned a
10-point loss into a 20-point gain. Same model, same candidates, different
input shape.

**Two metrics are not a comparison.** An earlier version of this evaluation
scored retrieval with a strict function in one script and a token-overlap
function in another, then compared the outputs. On a *total retrieval miss*,
the token-overlap rule awarded 0.40 to 1.00, because financial prose reuses the
same vocabulary everywhere. The resulting "reranking lifts recall from 0.502 to
0.994" was mostly a change of ruler. Everything is now scored by one rule in
one script.

**Hybrid retrieval trades top-3 precision for depth.** On its own, adding BM25
looked like a downgrade. It only pays off once a working reranker can reorder
the deeper pool — the two changes are worth more together than either alone.

**Citations need their own metric.** Page footers survive inside the chunk text
as bare digits, and in testing the model answered correctly while citing a page
number it had copied out of the body text rather than from any evidence label.
A faithfulness judge passes that, because the claim *is* supported. Citation
validity — every cited page must be one that was actually offered — is tracked
separately for exactly this reason.

---

## Limitations

- Ten answerable questions is a small benchmark. One question is worth 0.10, so
  a gap that size is a single item, not a trend.
- The corpus is one document. Nothing here tests retrieval across filings,
  companies, or reporting periods.
- Faithfulness is judged by a model. It checks grounding against the retrieved
  evidence, and will not catch an answer that is faithful to the wrong passage.
- The strict metric penalises correct retrievals whose gold span straddles a
  chunk boundary. It understates real usefulness, and is chosen for
  comparability rather than realism.
- The semantic chunker's size guard only splits *between* semantic units, so
  units longer than the 600-word ceiling pass through whole. Actual spread:
  172 chunks in band, 51 under, 35 over, range 10 to 2,436 words. This is the
  root cause of the reranker truncation problem above.
- Latency is measured warm, single-user, with no concurrency, and moves by
  roughly 0.2s between runs, so it is quoted to one decimal place. A cold start
  adds a few seconds more while the cross-encoder loads on first use.

---

## Repository layout

```
config.py                  models and paths, single source of truth
inspect_document.py        HTML -> cleaned text
page_map.py                printed page numbers from footer positions
build_fixed_chunks.py      fixed-size chunking
build_semantic_chunks.py   semantic chunking
build_vector_store.py      embed once, persist FAISS indexes
retrieval.py               dense / sparse / hybrid / reranking
rag_pipeline.py            answer with citations, refusal, latency
evaluate_retrieval.py      18 configurations, one scoring rule
evaluate_answers.py        faithfulness, refusal, citation validity, latency

data/raw/                  source filing and cleaned text
data/*.json                chunk sets
data/index/                FAISS indexes (regenerable, not committed)
evaluation/                benchmark questions, results, report
```

## Models

| Role | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | `gpt-4o-mini` (temperature 0) |
| Evaluation judge | `gpt-4o` (temperature 0) |

The judge is deliberately a different model from the generator, so the system
is not grading its own work.
