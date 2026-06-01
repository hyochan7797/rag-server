"""
Retrieval quality evaluation for the RAG pipeline.

Run after Qdrant has documents loaded:
  python eval_retrieval.py

Inside Docker:
  docker compose exec fastapi python eval_retrieval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any
from query_expansion import expand_domain_synonyms
from filter_extraction import extract_filters_from_query

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path: Path, override: bool = False) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value


ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN_SET = PYTHON_DIR / "eval_golden_set.json"


def load_environment() -> None:
    load_dotenv(ROOT_DIR / ".env", override=False)
    load_dotenv(PYTHON_DIR / ".env", override=False)
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("Golden set must be a JSON list.")
    return cases


def get_generation_model() -> Any | None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except ModuleNotFoundError:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


async def build_hyde_query(model: Any | None, question: str, search_query: str) -> str | None:
    if model is None:
        return None

    prompt = f"""금융 대출 상품 검색을 위한 HyDE 문서를 작성하세요.
아래 질문에 직접 답하는 것처럼, 실제 상품 문서에 들어 있을 법한 짧은 설명문을 만드세요.

[규칙]
1. 은행명, 대출종류, 상품명 후보, 금리, 한도, 가입조건 같은 검색 키워드를 자연스럽게 포함하세요.
2. 확정되지 않은 숫자를 지어내지 말고, 필요하면 "금리", "한도", "조건" 같은 일반 표현을 쓰세요.
3. 2~4문장으로만 작성하세요.
4. 설명문만 출력하세요.

[원래 질문]: {question}
[검색 최적화 쿼리]: {search_query}

HyDE 설명문:"""
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
    except Exception as exc:
        print(f"HyDE generation failed, falling back to normal Dense query: {exc}")
        return None
    return response.text.strip().strip('"').strip("'") or None


def first_relevant_rank(docs: list[str], expected_any: list[str]) -> int | None:
    needles = [value.lower() for value in expected_any if value]
    for idx, doc in enumerate(docs, start=1):
        haystack = doc.lower()
        if any(needle in haystack for needle in needles):
            return idx
    return None


def recall_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


async def evaluate_case(
    case: dict[str, Any],
    k: int,
    candidate_k: int,
    hyde_model: Any | None = None,
) -> dict[str, Any]:
    from rag_pipeline import search_similar_docs

    question = case["question"]
    search_query = expand_domain_synonyms(case.get("rewritten_query") or question)
    hyde_query = await build_hyde_query(hyde_model, question, search_query)
    auto_banks, auto_types = extract_filters_from_query(search_query)
    allowed_banks = case.get("allowed_banks", auto_banks)
    allowed_types = case.get("allowed_loan_types", auto_types)
    docs, scores = await search_similar_docs(
        history_list=[{"role": "user", "content": question}],
        query=question,
        allowed_banks=allowed_banks,
        allowed_types=allowed_types,
        rewritten_query=search_query,
        hyde_query=hyde_query,
        top_k=k,
        candidate_k=candidate_k,
    )
    rank = first_relevant_rank(docs, case.get("expected_any", []))
    return {
        "id": case.get("id", question),
        "difficulty": case.get("difficulty", "unspecified"),
        "question": question,
        "search_query": search_query,
        "hyde_query": hyde_query,
        "allowed_banks": allowed_banks,
        "allowed_loan_types": allowed_types,
        "rank": rank,
        "recall_at_1": recall_at(rank, 1),
        f"recall_at_{k}": recall_at(rank, k),
        "mrr": reciprocal_rank(rank),
        f"ndcg_at_{k}": ndcg_at(rank, k),
        "top_results": [
            {
                "rank": idx,
                "score": scores[idx - 1] if idx - 1 < len(scores) else None,
                "preview": doc.replace("\n", " ")[:180],
            }
            for idx, doc in enumerate(docs, start=1)
        ],
    }


def average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def filter_cases(cases: list[dict[str, Any]], difficulty: str) -> list[dict[str, Any]]:
    if difficulty == "all":
        return cases
    return [case for case in cases if case.get("difficulty") == difficulty]


def print_report(rows: list[dict[str, Any]], k: int) -> None:
    print("\nRetrieval Evaluation")
    print("=" * 72)
    for row in rows:
        rank = row["rank"] if row["rank"] is not None else "-"
        print(f"{row['id']:<36} rank={rank}  q={row['question']}")

    print("-" * 72)
    print(f"cases      : {len(rows)}")
    print(f"Recall@1   : {average(rows, 'recall_at_1'):.3f}")
    print(f"Recall@{k:<2}  : {average(rows, f'recall_at_{k}'):.3f}")
    print(f"MRR        : {average(rows, 'mrr'):.3f}")
    print(f"NDCG@{k:<2} : {average(rows, f'ndcg_at_{k}'):.3f}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--use-hyde", action="store_true")
    parser.add_argument("--hyde-delay", type=float, default=13.0)
    parser.add_argument("--difficulty", choices=["all", "easy", "hard"], default="all")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    load_environment()
    cases = filter_cases(load_cases(Path(args.golden_set)), args.difficulty)
    hyde_model = get_generation_model() if args.use_hyde else None
    if args.use_hyde and hyde_model is None:
        raise RuntimeError("HyDE requires GOOGLE_API_KEY and google-generativeai.")

    rows = []
    for idx, case in enumerate(cases):
        if args.use_hyde and idx > 0 and args.hyde_delay > 0:
            await asyncio.sleep(args.hyde_delay)
        rows.append(
            await evaluate_case(
                case,
                k=args.k,
                candidate_k=args.candidate_k,
                hyde_model=hyde_model,
            )
        )

    print_report(rows, k=args.k)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote detailed results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
