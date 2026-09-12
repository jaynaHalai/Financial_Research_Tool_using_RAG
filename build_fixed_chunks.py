"""Fixed-size chunking: 500-word windows with 50 words of overlap."""

import json
from pathlib import Path

from page_map import build_page_map, pages_for_span

CHUNK_SIZE = 500
OVERLAP = 50

text = Path("data/raw/arm_report.txt").read_text(encoding="utf-8")

word_offsets, page_numbers = build_page_map(text)

words = text.split()

chunks = []
step = CHUNK_SIZE - OVERLAP

for start in range(0, len(words), step):
    chunk_words = words[start:start + CHUNK_SIZE]

    if not chunk_words:
        break

    end = start + len(chunk_words)

    chunks.append({
        "chunk_id": len(chunks),
        "strategy": "fixed",
        "text": " ".join(chunk_words),
        "pages": pages_for_span(word_offsets, page_numbers, start, end),
    })

Path("data/fixed_chunks.json").write_text(
    json.dumps(chunks, indent=2),
    encoding="utf-8"
)

sizes = [len(chunk["text"].split()) for chunk in chunks]

print(f"Created {len(chunks)} fixed-size chunks.")
print(f"Average size: {sum(sizes) / len(sizes):.1f} words")
print(f"Size range: {min(sizes)} to {max(sizes)} words")
print(f"Pages covered: {chunks[0]['pages'][0]} to {chunks[-1]['pages'][-1]}")
