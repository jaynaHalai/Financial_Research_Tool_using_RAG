import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import CrossEncoder


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")


# Load semantic chunks.
with open("data/semantic_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0
)


chunk_texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(chunks)} chunks.")


# Ask the user for a research question.
question = input("\nAsk a financial research question: ")


# Retrieve the 30 most similar chunks.
query_vector = embeddings.embed_query(question)

scores = []

for i, vector in enumerate(
    embeddings.embed_documents(chunk_texts)
):
    score = sum(
        a * b
        for a, b in zip(query_vector, vector)
    )

    scores.append((score, i))

scores.sort(reverse=True)

candidates = scores[:30]


# Rerank the retrieved candidates.
pairs = [
    (question, chunks[index]["text"])
    for _, index in candidates
]

rerank_scores = reranker.predict(pairs)

reranked = sorted(
    zip(rerank_scores, candidates),
    reverse=True
)


# Keep the top 5 evidence chunks.
top_chunks = [
    chunks[index]["text"]
    for _, (_, index) in reranked[:5]
]


context = "\n\n---\n\n".join(top_chunks)


prompt = f"""
You are a financial research assistant.

Answer the user's question using only the evidence provided below.

If the evidence does not contain enough information to answer,
say that the evidence is insufficient.

Question:
{question}

Evidence:
{context}

Give a concise answer and do not invent facts.
"""


response = llm.invoke(prompt)


print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)
print(response.content)