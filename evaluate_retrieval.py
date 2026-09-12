import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")

embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

chunks = json.loads(
    Path("data/semantic_chunks.json").read_text(encoding="utf-8")
)

questions = json.loads(
    Path("evaluation/pilot_questions.json").read_text(encoding="utf-8")
)

chunk_texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(chunks)} fixed chunks.")
print(f"Loaded {len(questions)} evaluation questions.")

chunk_vectors = embeddings.embed_documents(chunk_texts)

for question in questions:
    query_vector = embeddings.embed_query(question["question"])

    scores = []

    for i, vector in enumerate(chunk_vectors):
        score = sum(a * b for a, b in zip(query_vector, vector))
        scores.append((score, i))

    scores.sort(reverse=True)

    print("\n" + "=" * 70)
    print(question["id"])
    print(question["question"])
    print("=" * 70)

    for rank, (score, chunk_index) in enumerate(scores[:10], start=1):
        print(f"\nRank {rank} | Score {score:.4f} | Chunk {chunk_index}")
        print(chunks[chunk_index]["text"][:500])