import json
from pathlib import Path

text = Path("data/raw/arm_report.txt").read_text(encoding="utf-8")

words = text.split()

chunk_size = 500
overlap = 50

chunks = []

step = chunk_size - overlap

for start in range(0, len(words), step):
    chunk_words = words[start:start + chunk_size]

    if not chunk_words:
        break

    chunks.append({
        "chunk_id": len(chunks),
        "text": " ".join(chunk_words)
    })

Path("data/fixed_chunks.json").write_text(
    json.dumps(chunks, indent=2),
    encoding="utf-8"
)

print(f"Created {len(chunks)} fixed-size chunks.")