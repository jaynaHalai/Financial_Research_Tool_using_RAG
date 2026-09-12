"""Single source of truth for models and paths.

These values were previously repeated across the build and query scripts. That
mattered most for the embedding model: if the index were built with one model
and queried with another, retrieval would degrade silently rather than fail,
and nothing in the output would say why.

Deliberately free of heavy imports, so the chunk builders can read it without
pulling in FAISS or the cross-encoder.
"""

from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"

SOURCE_HTML = Path("data/raw/SEC Filing – Arm®.html")
CLEANED_TEXT = Path("data/raw/arm_report.txt")

CHUNK_PATHS = {
    "fixed": Path("data/fixed_chunks.json"),
    "semantic": Path("data/semantic_chunks.json"),
}

INDEX_DIR = Path("data/index")

QUESTIONS = Path("evaluation/pilot_questions.json")
RETRIEVAL_RESULTS = Path("evaluation/retrieval_results.json")
ANSWER_RESULTS = Path("evaluation/answer_results.json")
