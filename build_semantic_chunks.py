import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")

text = Path("data/raw/arm_report.txt").read_text(encoding="utf-8")

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

target_words = 500
min_words = 400
max_words = 600

chunks = []
current = []

for doc in documents:
    words = doc.page_content.split()

    if current and len(current) + len(words) > max_words:
        chunks.append({
            "chunk_id": len(chunks),
            "text": " ".join(current)
        })
        current = []

    current.extend(words)

if current:
    chunks.append({
        "chunk_id": len(chunks),
        "text": " ".join(current)
    })

Path("data/semantic_chunks.json").write_text(
    json.dumps(chunks, indent=2),
    encoding="utf-8"
)

sizes = [len(chunk["text"].split()) for chunk in chunks]

print(f"Created {len(chunks)} semantic chunks.")
print(f"Average size: {sum(sizes) / len(sizes):.1f} words")
print(f"Smallest chunk: {min(sizes)} words")
print(f"Largest chunk: {max(sizes)} words")