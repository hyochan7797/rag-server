from __future__ import annotations

import re
from typing import Optional


BANK_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"국민\s*은행|kb\s*국민|kb\s*은행|^kb\b", re.IGNORECASE), ["국민은행"]),
    (re.compile(r"우리\s*은행", re.IGNORECASE), ["우리은행"]),
    (re.compile(r"신한\s*은행", re.IGNORECASE), ["신한은행"]),
    (re.compile(r"하나\s*은행", re.IGNORECASE), ["하나은행", "주식회사 하나은행"]),
    (re.compile(r"농협|nh\s*농협|농협\s*은행", re.IGNORECASE), ["농협은행주식회사"]),
    (re.compile(r"한화\s*손해\s*보험", re.IGNORECASE), ["한화손해보험주식회사"]),
    (re.compile(r"kb\s*자산\s*운용", re.IGNORECASE), ["KB자산운용"]),
    (re.compile(r"현대\s*카드", re.IGNORECASE), ["현대카드㈜"]),
    (re.compile(r"푸본\s*현대\s*생명", re.IGNORECASE), ["푸본현대생명보험주식회사"]),
    (re.compile(r"웰컴\s*저축\s*은행", re.IGNORECASE), ["웰컴저축은행"]),
]


TYPE_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"신용\s*대출|개인\s*신용|마이너스\s*통장|마통|한도\s*대출", re.IGNORECASE), ["sinyoung"]),
    (re.compile(r"주택\s*담보|주담대|아파트\s*담보|아담대", re.IGNORECASE), ["dambo_mortgage"]),
    (re.compile(r"전세\s*자금|전세\s*대출", re.IGNORECASE), ["dambo_jeonse"]),
    (re.compile(r"정기\s*예금|예금", re.IGNORECASE), ["deposit"]),
    (re.compile(r"적금|월복리|자유\s*적금", re.IGNORECASE), ["saving"]),
    (re.compile(r"연금\s*저축|연금\s*펀드|수령액|10년\s*확정", re.IGNORECASE), ["annuity_saving"]),
    (re.compile(r"금융\s*회사|은행\s*정보|대표\s*전화|전화번호|홈페이지|고객\s*센터|영업\s*여부|지역별", re.IGNORECASE), ["company"]),
]


def _add_many(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def extract_filters_from_query(query: str) -> tuple[Optional[list[str]], Optional[list[str]]]:
    extracted_banks: list[str] = []
    extracted_types: list[str] = []

    for pattern, bank_names in BANK_PATTERNS:
        if pattern.search(query):
            _add_many(extracted_banks, bank_names)

    for pattern, product_types in TYPE_PATTERNS:
        if pattern.search(query):
            _add_many(extracted_types, product_types)

    if (
        re.search(r"\bkb\b|kb\s*", query, re.IGNORECASE)
        and re.search(r"연금\s*저축|연금\s*펀드|펀드|자산\s*운용|가치주", query, re.IGNORECASE)
    ):
        _add_many(extracted_banks, ["KB자산운용"])

    return extracted_banks or None, extracted_types or None
