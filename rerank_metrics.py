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
    Path("data/fixed_chunks.json").read_text(encoding="utf-8")
)

questions = json.loads(
    Path("evaluation/pilot_questions.json").read_text(encoding="utf-8")
)

chunk_texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(chunks)} semantic chunks.")
print(f"Loaded {len(questions)} evaluation questions.")

chunk_vectors = embeddings.embed_documents(chunk_texts)


def normalize(text):
    return " ".join(text.lower().split())


def coverage(gold, retrieved_text):
    gold = normalize(gold)
    retrieved = normalize(retrieved_text)

    if gold in retrieved:
        return 1.0

    gold_tokens = gold.split()

    if not gold_tokens:
        return 0.0

    matched = sum(
        token in retrieved
        for token in gold_tokens
    )

    return matched / len(gold_tokens)


results = {3: [], 5: [], 10: []}

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

    for k in [3, 5, 10]:
        retrieved = " ".join(
            chunks[index]["text"]
            for _, (_, index) in reranked[:k]
        )

        span_scores = []

        for evidence in question["required_evidence"]:
            score = coverage(
                evidence["text"],
                retrieved
            )

            span_scores.append(score)

        question_recovery = (
            sum(span_scores) / len(span_scores)
        )

        all_required = all(
            score >= 0.95
            for score in span_scores
        )

        results[k].append({
            "id": question["id"],
            "recovery": question_recovery,
            "all_required": all_required
        })


print("\n" + "=" * 70)
print("Fixed + RERANKING RESULTS")
print("=" * 70)

for k in [3, 5, 10]:
    average_recovery = (
        sum(item["recovery"] for item in results[k])
        / len(results[k])
    )

    all_required_rate = (
        sum(item["all_required"] for item in results[k])
        / len(results[k])
    )

    print(f"\nRecall@{k}")
    print(f"Required-span recovery: {average_recovery:.3f}")
    print(f"All-required recovery:  {all_required_rate:.3f}")