"""Retrieval evaluation across every chunking and retrieval configuration.

One scoring rule is used everywhere: a required span counts as recovered only
if it appears verbatim (whitespace-normalised) in the concatenated retrieved
text. Partial-credit scoring was tried first and rejected, because token-overlap
scoring awarded 0.4 to 1.0 to retrievals that contained none of the gold text.

Metrics, over the answerable questions only:
  span recovery   - mean fraction of a question's required spans recovered
  full recovery   - fraction of questions where every required span was found
"""

import json
import time
from pathlib import Path

from retrieval import Retriever

STRATEGIES = ["fixed", "semantic"]
MODES = ["dense", "sparse", "hybrid"]
# (label suffix, reranker on?, window-level?)
RERANK = [
    ("", False, False),
    (" + rerank(truncated)", True, False),
    (" + rerank(windowed)", True, True),
]
K_VALUES = [3, 5, 10]
CANDIDATES = 20


def normalize(text):
    return " ".join(text.lower().split())


def recovered(span, retrieved_text):
    """Strict containment: the span is either present verbatim or it is not."""
    return 1.0 if normalize(span) in retrieved_text else 0.0


questions = json.loads(
    Path("evaluation/pilot_questions.json").read_text(encoding="utf-8")
)

answerable = [q for q in questions if q["answerable"]]

print(f"Evaluating {len(answerable)} answerable questions "
      f"({len(questions) - len(answerable)} unanswerable held out for "
      f"evaluate_answers.py).")

rows = []

for strategy in STRATEGIES:
    retriever = Retriever(strategy)

    for mode in MODES:
        for suffix, use_reranker, windowed in RERANK:
            label = f"{strategy} + {mode}{suffix}"

            per_k = {k: {"span": [], "full": []} for k in K_VALUES}
            latencies = []

            for question in answerable:
                started = time.perf_counter()

                hits = retriever.search(
                    question["question"],
                    mode=mode,
                    candidates=CANDIDATES,
                    top_k=max(K_VALUES),
                    use_reranker=use_reranker,
                    windowed_rerank=windowed,
                )

                latencies.append(time.perf_counter() - started)

                for k in K_VALUES:
                    retrieved_text = normalize(" ".join(
                        retriever.chunks[chunk_id]["text"]
                        for chunk_id, _ in hits[:k]
                    ))

                    scores = [
                        recovered(evidence["text"], retrieved_text)
                        for evidence in question["required_evidence"]
                    ]

                    per_k[k]["span"].append(sum(scores) / len(scores))
                    per_k[k]["full"].append(1.0 if all(scores) else 0.0)

            row = {
                "configuration": label,
                "strategy": strategy,
                "mode": mode,
                "reranked": use_reranker,
                "windowed_rerank": windowed,
                "median_retrieval_seconds": round(
                    sorted(latencies)[len(latencies) // 2], 3
                ),
            }

            for k in K_VALUES:
                row[f"span_recovery_at_{k}"] = round(
                    sum(per_k[k]["span"]) / len(per_k[k]["span"]), 3
                )
                row[f"full_recovery_at_{k}"] = round(
                    sum(per_k[k]["full"]) / len(per_k[k]["full"]), 3
                )

            rows.append(row)
            print(f"  done: {label}")

results = {
    "benchmark": "Arm Holdings FY2025 Financial Research RAG",
    "answerable_questions": len(answerable),
    "candidates_before_rerank": CANDIDATES,
    "metric": "strict verbatim span containment in retrieved text",
    "results": rows,
}

Path("evaluation/retrieval_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

header = f'{"configuration":<34}{"span@3":>8}{"span@5":>8}{"span@10":>9}{"full@5":>8}{"sec":>7}'

print("\n" + "=" * len(header))
print("RETRIEVAL RESULTS")
print("=" * len(header))
print(header)
print("-" * len(header))

for row in sorted(rows, key=lambda r: -r["span_recovery_at_5"]):
    print(
        f'{row["configuration"]:<34}'
        f'{row["span_recovery_at_3"]:>8.3f}'
        f'{row["span_recovery_at_5"]:>8.3f}'
        f'{row["span_recovery_at_10"]:>9.3f}'
        f'{row["full_recovery_at_5"]:>8.3f}'
        f'{row["median_retrieval_seconds"]:>7.2f}'
    )

print("\nWritten to evaluation/retrieval_results.json")
