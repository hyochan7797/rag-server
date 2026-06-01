from __future__ import annotations


def _add_alias(aliases: list[str], value: str) -> None:
    if value and value not in aliases:
        aliases.append(value)


def build_product_aliases(product_name: str, loan_type: str) -> list[str]:
    """Build stable search aliases from product metadata at indexing time."""
    aliases: list[str] = []
    name = product_name or ""

    if loan_type == "sinyoung":
        _add_alias(aliases, "신용대출")
        _add_alias(aliases, "일반신용대출")

    if loan_type.startswith("dambo"):
        _add_alias(aliases, "담보대출")

    if loan_type == "dambo_mortgage":
        _add_alias(aliases, "주택담보대출")
        _add_alias(aliases, "주담대")
        _add_alias(aliases, "아파트담보대출")
        _add_alias(aliases, "아담대")

    if loan_type == "dambo_jeonse":
        _add_alias(aliases, "전세자금대출")
        _add_alias(aliases, "전세대출")

    if "마이너스" in name or "한도대출" in name:
        _add_alias(aliases, "마통")
        _add_alias(aliases, "마이너스통장")
        _add_alias(aliases, "마이너스한도대출")
        _add_alias(aliases, "한도대출")

    if "전세" in name:
        _add_alias(aliases, "전세대출")
        _add_alias(aliases, "전세자금대출")

    if "아파트" in name:
        _add_alias(aliases, "아파트담보대출")
        _add_alias(aliases, "아담대")

    if "주택담보" in name:
        _add_alias(aliases, "주택담보대출")
        _add_alias(aliases, "주담대")

    if loan_type == "deposit":
        _add_alias(aliases, "정기예금")
        _add_alias(aliases, "예금")
        _add_alias(aliases, "금리 비교")

    if loan_type == "saving":
        _add_alias(aliases, "적금")
        _add_alias(aliases, "정기적금")
        _add_alias(aliases, "자유적금")
        _add_alias(aliases, "자유적립식")

    if loan_type == "saving" and ("1934" in name or "청년" in name):
        _add_alias(aliases, "청년")
        _add_alias(aliases, "청년적금")
        _add_alias(aliases, "만19세")
        _add_alias(aliases, "만34세")

    if loan_type == "saving" and "월복리" in name:
        _add_alias(aliases, "월복리")
        _add_alias(aliases, "월복리적금")

    if loan_type in {"deposit", "saving"}:
        _add_alias(aliases, "모바일")
        _add_alias(aliases, "앱")
        _add_alias(aliases, "스마트폰")

    if loan_type == "annuity_saving":
        _add_alias(aliases, "연금저축")
        _add_alias(aliases, "연금")

    if loan_type == "company":
        _add_alias(aliases, "금융회사")
        _add_alias(aliases, "은행정보")

    return aliases
