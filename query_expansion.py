from __future__ import annotations

import re


CORE_QUERY_SYNONYMS: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"\b마통\b|마이너스\s*통장", re.IGNORECASE),
        ["마이너스통장", "마이너스한도대출"],
    ),
    (
        re.compile(r"주담대", re.IGNORECASE),
        ["주택담보대출"],
    ),
    (
        re.compile(r"아담대", re.IGNORECASE),
        ["아파트담보대출"],
    ),
    (
        re.compile(r"전세\s*대출", re.IGNORECASE),
        ["전세자금대출"],
    ),
    (
        re.compile(r"정기\s*예금", re.IGNORECASE),
        ["예금", "정기예금"],
    ),
    (
        re.compile(r"연금\s*저축", re.IGNORECASE),
        ["연금저축"],
    ),
    (
        re.compile(r"\b앱\b|모바일", re.IGNORECASE),
        ["스마트폰"],
    ),
    (
        re.compile(r"청년", re.IGNORECASE),
        ["만19세", "만34세", "청년적금"],
    ),
    (
        re.compile(r"월\s*복리", re.IGNORECASE),
        ["월복리", "월복리적금"],
    ),
]


def expand_domain_synonyms(query: str) -> str:
    """Append only high-confidence finance abbreviations at query time."""
    additions: list[str] = []
    normalized_query = query.lower()

    for pattern, synonyms in CORE_QUERY_SYNONYMS:
        if not pattern.search(query):
            continue
        for synonym in synonyms:
            if synonym.lower() not in normalized_query and synonym not in additions:
                additions.append(synonym)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
