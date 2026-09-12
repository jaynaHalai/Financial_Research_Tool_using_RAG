"""End-to-end answer evaluation: faithfulness, refusal behaviour, latency.

Retrieval quality is measured separately in evaluate_retrieval.py. This script
measures what the user actually receives:

  faithfulness     - is every claim in the answer supported by the evidence
                     that was retrieved for it (judged by a second model)
  correct refusal  - unanswerable questions that were refused
  false refusal    - answerable questions that were refused anyway
  citation rate    - answers that cite at least one page
  citation validity- every page cited was actually one of the pages offered
                     as evidence. Page footers survive inside the chunk text,
                     so a model can copy a stray number and cite a page it was
                     never shown; this metric catches that.

The judge is a different model from the generator, so the system is not
grading its own work.
"""

import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import ANSWER_RESULTS, GENERATION_MODEL, JUDGE_MODEL, QUESTIONS
from rag_pipeline import answer, build, format_evidence, invoke_with_retry

JUDGE_PROMPT = """You are auditing a financial research assistant for \
faithfulness.

Below is the evidence the assistant was given, and the answer it produced.

Decide whether EVERY factual claim in the answer is directly supported by the \
evidence. Ignore style, completeness and whether the answer is helpful. Judge \
only whether it is grounded.

Reply with exactly one word, FAITHFUL or UNFAITHFUL, then a newline, then one \
short sentence of justification.

Evidence:
{evidence}

Answer:
{answer}"""


def offered_pages(citations):
    """Expand evidence labels such as "pages 124-132" into a set of pages."""
    allowed = set()

    for label in citations:
        numbers = [int(n) for n in re.findall(r"\d+", label)]

        if len(numbers) == 2:
            allowed.update(range(numbers[0], numbers[1] + 1))
        elif numbers:
            allowed.add(numbers[0])

    return allowed


def cited_pages(text):
    """Pages the answer actually cites."""
    pages = set()

    for match in re.finditer(r"pages?\s+(\d+)\s*(?:-\s*(\d+))?", text, re.I):
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) else first
        pages.update(range(first, last + 1))

    return pages


def main():
    load_dotenv()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    retriever, llm = build()
    judge = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=JUDGE_MODEL,
        temperature=0,
    )

    records = []

    for question in questions:
        result = answer(question["question"], retriever, llm)

        result["id"] = question["id"]
        result["answerable"] = question["answerable"]
        claimed = cited_pages(result["answer"])
        allowed = offered_pages(result["citations"])

        result["cited"] = bool(claimed)
        result["cited_pages"] = sorted(claimed)
        result["unsupported_citations"] = sorted(claimed - allowed)
        result["citations_valid"] = not (claimed - allowed)

        # Only grounded-ness of a real answer is worth judging; a refusal
        # asserts nothing and cannot be unfaithful.
        if result["refused"]:
            result["faithful"] = None
            result["judge_note"] = "refused, not judged"
        else:
            verdict = invoke_with_retry(
                judge,
                JUDGE_PROMPT.format(
                    evidence=format_evidence(retriever, result["chunk_ids"]),
                    answer=result["answer"],
                ),
            ).content.strip()

            result["faithful"] = verdict.upper().startswith("FAITHFUL")
            result["judge_note"] = " ".join(verdict.split("\n")[1:]).strip()

        records.append(result)

        status = "REFUSED" if result["refused"] else (
            "faithful" if result["faithful"] else "UNFAITHFUL"
        )
        print(f'  {question["id"]}  {status:<11} '
              f'{result["total_seconds"]}s')

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    judged = [r for r in answerable if r["faithful"] is not None]

    faithfulness = (
        sum(r["faithful"] for r in judged) / len(judged) if judged else 0.0
    )
    correct_refusal = (
        sum(r["refused"] for r in unanswerable) / len(unanswerable)
    )
    false_refusal = sum(r["refused"] for r in answerable) / len(answerable)
    citation_rate = (
        sum(r["cited"] for r in judged) / len(judged) if judged else 0.0
    )
    citation_validity = (
        sum(r["citations_valid"] for r in judged) / len(judged)
        if judged else 0.0
    )

    latencies = sorted(r["total_seconds"] for r in records)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]

    summary = {
        "generation_model": GENERATION_MODEL,
        "judge_model": JUDGE_MODEL,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "faithfulness": round(faithfulness, 3),
        "correct_refusal_rate": round(correct_refusal, 3),
        "false_refusal_rate": round(false_refusal, 3),
        "citation_rate": round(citation_rate, 3),
        "citation_validity": round(citation_validity, 3),
        "latency_p50_seconds": p50,
        "latency_p95_seconds": p95,
    }

    ANSWER_RESULTS.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 58)
    print("ANSWER QUALITY")
    print("=" * 58)
    print(f'Faithfulness (of {len(judged)} answered)   '
          f'{summary["faithfulness"]:.0%}')
    print(f'Correct refusal (unanswerable)     '
          f'{summary["correct_refusal_rate"]:.0%}')
    print(f'False refusal (answerable)         '
          f'{summary["false_refusal_rate"]:.0%}')
    print(f'Citation rate                      '
          f'{summary["citation_rate"]:.0%}')
    print(f'Citation validity                  '
          f'{summary["citation_validity"]:.0%}')
    print(f'Latency p50 / p95                  {p50}s / {p95}s')
    print("\nWritten to evaluation/answer_results.json")


if __name__ == "__main__":
    main()
