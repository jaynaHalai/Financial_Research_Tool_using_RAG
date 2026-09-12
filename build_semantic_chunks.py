"""Semantic chunking: split on embedding-similarity breakpoints.

Note the known limitation reported at the end of this script: the max-size
guard only splits *between* semantic units, so any single unit longer than
MAX_WORDS passes through whole. The resulting size spread is what the
chunking comparison in evaluate_retrieval.py measures against fixed chunks.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from page_map import build_page_map, pages_for_span

MIN_WORDS = 400
MAX_WORDS = 600

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")

text = Path("data/raw/arm_report.txt").read_text(encoding="utf-8")

word_offsets, page_numbers = build_page_map(text)

embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80
)

documents = splitter.create_documents([text])

chunks = []
current = []
current_start = 0
consumed = 0


def flush(words, start):
    chunks.append({
        "chunk_id": len(chunks),
        "strategy": "semantic",
        "text": " ".join(words),
        "pages": pages_for_span(
            word_offsets, page_numbers, start, start + len(words)
        ),
    })


for document in documents:
    words = document.page_content.split()

    if current and len(current) + len(words) > MAX_WORDS:
        flush(current, current_start)
        current = []
        current_start = consumed

    current.extend(words)
    consumed += len(words)

if current:
    flush(current, current_start)

Path("data/semantic_chunks.json").write_text(
    json.dumps(chunks, indent=2),
    encoding="utf-8"
)

sizes = [len(chunk["text"].split()) for chunk in chunks]

oversized = [size for size in sizes if size > MAX_WORDS]
undersized = [size for size in sizes if size < MIN_WORDS]
in_band = len(sizes) - len(oversized) - len(undersized)

print(f"Created {len(chunks)} semantic chunks.")
print(f"Average size: {sum(sizes) / len(sizes):.1f} words")
print(f"Size range: {min(sizes)} to {max(sizes)} words")
print()
print("Size distribution (the known limitation, for the report):")
print(f"  Within {MIN_WORDS}-{MAX_WORDS} words: {in_band}")
print(f"  Over {MAX_WORDS} words:          {len(oversized)}")
print(f"  Under {MIN_WORDS} words:         {len(undersized)}")
