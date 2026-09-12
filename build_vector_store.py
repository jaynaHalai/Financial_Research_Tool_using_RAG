"""Embed each chunk set once and persist a FAISS index to disk.

Embedding happens here, not at query time, so answering a question costs one
embedding call for the query instead of one for every chunk in the corpus.
"""

import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config import CHUNK_PATHS, EMBEDDING_MODEL, INDEX_DIR

BATCH_SIZE = 64

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")

embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBEDDING_MODEL)

INDEX_DIR.mkdir(parents=True, exist_ok=True)

for strategy, chunk_path in CHUNK_PATHS.items():
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    texts = [chunk["text"] for chunk in chunks]

    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))

        print(
            f"  {strategy}: embedded "
            f"{min(start + BATCH_SIZE, len(texts))}/{len(texts)}",
            end="\r"
        )

    matrix = np.array(vectors, dtype="float32")

    # Normalise so inner product is cosine similarity.
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_DIR / f"{strategy}.faiss"))

    print(
        f"  {strategy}: {index.ntotal} vectors, "
        f"dimension {matrix.shape[1]}          "
    )

manifest = {
    "embedding_model": EMBEDDING_MODEL,
    "strategies": list(CHUNK_PATHS),
    "similarity": "cosine (L2-normalised inner product)",
}

(INDEX_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print(f"\nIndexes written to {INDEX_DIR}/")
