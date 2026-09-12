import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")

embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

chunks = json.loads(
    Path("data/semantic_chunks.json").read_text(encoding="utf-8")
)

questions = json.loads(
    Path("evaluation/pilot_questions.json").read_text(encoding="utf-8")
)

chunk_texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(chunks)} chunks.")
print(f"Loaded {len(questions)} evaluation questions.")

chunk_vectors = embeddings.embed_documents(chunk_texts)

for question in questions:
    query = question["question"]

    query_vector = embeddings.embed_query(query)

    scores = []

    for i, vector in enumerate(chunk_vectors):
        score = sum(
            a * b
            for a, b in zip(query_vector, vector)
        )

        scores.append((score, i))

    scores.sort(reverse=True)

    # Retrieve 30 candidates before reranking.
    candidates = scores[:30]

    candidate_texts = [
        chunks[index]["text"]
        for _, index in candidates
    ]

    pairs = [
        (query, text)
        for text in candidate_texts
    ]

    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(rerank_scores, candidates),
        reverse=True
    )

    print("\n" + "=" * 70)
    print(question["id"])
    print(query)
    print("=" * 70)

    for rank, (rerank_score, (_, index)) in enumerate(
        reranked[:10],
        start=1
    ):
        print(
            f"\nRank {rank} | "
            f"Chunk {index} | "
            f"Score {rerank_score:.4f}"
        )

        print(chunks[index]["text"][:500])