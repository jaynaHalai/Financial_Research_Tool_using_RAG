"""Answer questions about the Arm FY2025 filing, with page citations.

Retrieval is hybrid (dense + BM25) followed by window-level cross-encoder
reranking; see retrieval.py. The generator is instructed to answer only from
the retrieved evidence and to refuse outright when the evidence does not
support an answer, because a wrong number in a financial answer is worse than
no answer.
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import GENERATION_MODEL
from retrieval import Retriever

MAX_ATTEMPTS = 4
DEFAULT_STRATEGY = "semantic"
DEFAULT_MODE = "hybrid"
TOP_K = 5
CANDIDATES = 20

REFUSAL = "I could not find this in the Arm FY2025 filing."

PROMPT = """You are a financial research assistant answering questions about \
Arm Holdings plc's FY2025 annual report.

Use ONLY the evidence below. Each piece of evidence begins with a bracketed \
page reference such as [page 125] or [pages 68-69].

Every factual claim must cite the page reference of the evidence it came from, \
copied exactly as shown, for example: "Total revenue was $4,007 million \
[page 125]." Never cite evidence by number; always cite the page.

If the evidence does not contain enough information to answer the question, \
reply with exactly this sentence and nothing else:
{refusal}

Do not use outside knowledge. Do not guess. Do not infer figures that are not \
stated.

Question:
{question}

Evidence:
{evidence}

Answer concisely, in no more than four sentences."""


def invoke_with_retry(model, prompt, max_attempts=MAX_ATTEMPTS):
    """Call the model, retrying transient API failures with backoff.

    A dropped connection part-way through an evaluation run would otherwise
    discard every result computed so far.
    """
    for attempt in range(max_attempts):
        try:
            return model.invoke(prompt)
        except Exception as error:
            if attempt == max_attempts - 1:
                raise

            wait = 2 ** attempt
            print(f"    API error ({error.__class__.__name__}), "
                  f"retrying in {wait}s")
            time.sleep(wait)


def format_evidence(retriever, chunk_ids):
    """Render retrieved chunks as page-labelled evidence blocks."""
    blocks = []

    for chunk_id in chunk_ids:
        blocks.append(
            f"[{retriever.cite(chunk_id)}]\n"
            f"{retriever.chunks[chunk_id]['text']}"
        )

    return "\n\n---\n\n".join(blocks)


def answer(question, retriever, llm, mode=DEFAULT_MODE, top_k=TOP_K):
    """Return the answer plus its evidence and a latency breakdown."""
    started = time.perf_counter()

    hits = retriever.search(
        question,
        mode=mode,
        candidates=CANDIDATES,
        top_k=top_k,
        use_reranker=True,
    )

    retrieved_at = time.perf_counter()

    if not hits:
        return {
            "question": question,
            "answer": REFUSAL,
            "refused": True,
            "citations": [],
            "chunk_ids": [],
            "retrieval_seconds": round(retrieved_at - started, 3),
            "generation_seconds": 0.0,
            "total_seconds": round(retrieved_at - started, 3),
        }

    prompt = PROMPT.format(
        refusal=REFUSAL,
        question=question,
        evidence=format_evidence(retriever, [cid for cid, _ in hits]),
    )

    response = invoke_with_retry(llm, prompt)
    finished = time.perf_counter()

    text = response.content.strip()

    return {
        "question": question,
        "answer": text,
        "refused": REFUSAL.lower().rstrip(".") in text.lower(),
        "citations": [retriever.cite(chunk_id) for chunk_id, _ in hits],
        "chunk_ids": [chunk_id for chunk_id, _ in hits],
        "retrieval_seconds": round(retrieved_at - started, 3),
        "generation_seconds": round(finished - retrieved_at, 3),
        "total_seconds": round(finished - started, 3),
    }


def build(strategy=DEFAULT_STRATEGY):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found.")

    retriever = Retriever(strategy)

    llm = ChatOpenAI(
        api_key=api_key, model=GENERATION_MODEL, temperature=0
    )

    return retriever, llm


def main():
    retriever, llm = build()

    print(f"Arm FY2025 filing: {len(retriever.chunks)} "
          f"{retriever.strategy} chunks indexed.")
    print("Ask a question, or press Enter to quit.\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            break

        result = answer(question, retriever, llm)

        print("\n" + "=" * 70)
        print(result["answer"])
        print("=" * 70)
        print(f"Evidence: {', '.join(result['citations'])}")
        print(
            f"Latency: {result['total_seconds']}s "
            f"(retrieval {result['retrieval_seconds']}s, "
            f"generation {result['generation_seconds']}s)\n"
        )


if __name__ == "__main__":
    main()
