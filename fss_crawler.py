import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from document_aliases import build_product_aliases

load_dotenv("/app/.env")

FSS_BASE_URL = "https://finlife.fss.or.kr/finlifeapi"
FSS_API_KEY = os.getenv("FSS_API_KEY")
BANK_GROUP_CODE = "020000"
ALL_GROUP_CODES = ["020000", "030200", "030300", "050000", "060000"]

PRODUCT_ENDPOINTS = {
    "sinyoung": "creditLoanProductsSearch",
    "dambo_mortgage": "mortgageLoanProductsSearch",
    "dambo_jeonse": "rentHouseLoanProductsSearch",
    "deposit": "depositProductsSearch",
    "saving": "savingProductsSearch",
    "annuity_saving": "annuitySavingProductsSearch",
    "company": "companySearch",
}

# Most product endpoints are bank-group scoped. Annuity saving and company search
# require a group code too, but they need broader sectors than bank-only.
PRODUCT_GROUP_CODES: Dict[str, List[Optional[str]]] = {
    "sinyoung": [BANK_GROUP_CODE],
    "dambo_mortgage": [BANK_GROUP_CODE],
    "dambo_jeonse": [BANK_GROUP_CODE],
    "deposit": [BANK_GROUP_CODE],
    "saving": [BANK_GROUP_CODE],
    "annuity_saving": ALL_GROUP_CODES,
    "company": ALL_GROUP_CODES,
}

PRODUCT_TYPE_KO = {
    "sinyoung": "개인신용대출",
    "dambo_mortgage": "주택담보대출",
    "dambo_jeonse": "전세자금대출",
    "deposit": "정기예금",
    "saving": "적금",
    "annuity_saving": "연금저축",
    "company": "금융회사",
}


async def _fetch_page(
    client: httpx.AsyncClient,
    endpoint: str,
    page_no: int,
    top_fin_grp_no: Optional[str] = BANK_GROUP_CODE,
) -> dict:
    url = f"{FSS_BASE_URL}/{endpoint}.json"
    params = {
        "auth": FSS_API_KEY,
        "pageNo": page_no,
    }
    if top_fin_grp_no:
        params["topFinGrpNo"] = top_fin_grp_no

    resp = await client.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    try:
        return json.loads(resp.content.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return resp.json()


async def _fetch_all_pages(
    endpoint: str,
    top_fin_grp_no: Optional[str] = BANK_GROUP_CODE,
) -> Tuple[list, list]:
    all_base, all_options = [], []
    async with httpx.AsyncClient() as client:
        page = 1
        while True:
            try:
                data = await _fetch_page(client, endpoint, page, top_fin_grp_no)
            except httpx.HTTPStatusError as e:
                print(f"  HTTP {e.response.status_code} error (page {page})")
                break

            result = data.get("result", {})
            err_cd = str(result.get("err_cd", ""))
            if err_cd and err_cd != "000":
                print(f"  API error {err_cd}: {result.get('err_msg', '')}")
                break

            base_list = result.get("baseList", [])
            opt_list = result.get("optionList", [])
            if not base_list:
                break

            all_base.extend(base_list)
            all_options.extend(opt_list)

            max_page = int(result.get("max_page_no", 1))
            now_page = int(result.get("now_page_no", page))
            if now_page >= max_page:
                break
            page += 1
            await asyncio.sleep(0.3)

    return all_base, all_options


async def _fetch_product_pages(product_type: str, endpoint: str) -> Tuple[list, list]:
    all_base, all_options = [], []
    seen_base = set()
    seen_options = set()

    for group_code in PRODUCT_GROUP_CODES.get(product_type, [BANK_GROUP_CODE]):
        label = group_code or "no topFinGrpNo"
        print(f"  - group={label}")
        base_list, option_list = await _fetch_all_pages(endpoint, group_code)

        for base in base_list:
            key = (
                base.get("dcls_month", ""),
                base.get("fin_co_no", ""),
                base.get("fin_prdt_cd", ""),
                base.get("kor_co_nm", ""),
            )
            if key in seen_base:
                continue
            seen_base.add(key)
            all_base.append(base)

        for option in option_list:
            key = tuple(sorted((k, str(v)) for k, v in option.items()))
            if key in seen_options:
                continue
            seen_options.add(key)
            all_options.append(option)

    return all_base, all_options


def _join_present(values: list[str], sep: str = " / ") -> str:
    return sep.join(str(v) for v in values if v not in (None, ""))


def _format_options_loan(options: list) -> str:
    if not options:
        return "금리 옵션 정보 없음"

    lines = []
    for opt in options:
        label = _join_present([
            opt.get("mrtg_type_nm", ""),
            opt.get("rpay_type_nm", ""),
            opt.get("lend_rate_type_nm", ""),
        ])
        line = f"  - {label}: {opt.get('lend_rate_min', '')}%~{opt.get('lend_rate_max', '')}%"
        if opt.get("lend_rate_avg", ""):
            line += f" (평균 {opt.get('lend_rate_avg')}%)"
        lines.append(line)
    return "\n".join(lines)


def _format_options_credit(options: list) -> str:
    if not options:
        return "금리 옵션 정보 없음"

    lines = []
    for opt in options:
        label = _join_present([
            opt.get("crdt_prdt_type_nm", ""),
            opt.get("crdt_lend_rate_type_nm", ""),
        ])
        avg = opt.get("crdt_grad_avg", "")
        grades = []
        for grade in range(1, 11):
            val = opt.get(f"crdt_grad_{grade}", "")
            if val:
                grades.append(f"{grade}등급 {val}%")

        line = f"  - {label}"
        if avg:
            line += f": 평균 {avg}%"
        if grades:
            line += f"\n    [{', '.join(grades)}]"
        lines.append(line)
    return "\n".join(lines)


def _format_options_deposit(options: list) -> str:
    if not options:
        return "금리 옵션 정보 없음"

    lines = []
    for opt in options:
        line = (
            f"  - save_trm={opt.get('save_trm', '')}개월"
            f" / intr_rate_type={opt.get('intr_rate_type', '')}"
            f" / intr_rate_type_nm={opt.get('intr_rate_type_nm', '')}"
            f" / intr_rate={opt.get('intr_rate', '')}%"
            f" / intr_rate2={opt.get('intr_rate2', '')}%"
        )
        lines.append(line)
    return "\n".join(lines)


def _format_options_saving(options: list) -> str:
    if not options:
        return "금리 옵션 정보 없음"

    lines = []
    for opt in options:
        line = (
            f"  - save_trm={opt.get('save_trm', '')}개월"
            f" / rsrv_type={opt.get('rsrv_type', '')}"
            f" / rsrv_type_nm={opt.get('rsrv_type_nm', '')}"
            f" / intr_rate_type={opt.get('intr_rate_type', '')}"
            f" / intr_rate_type_nm={opt.get('intr_rate_type_nm', '')}"
            f" / intr_rate={opt.get('intr_rate', '')}%"
            f" / intr_rate2={opt.get('intr_rate2', '')}%"
        )
        lines.append(line)
    return "\n".join(lines)


def _format_options_annuity(options: list) -> str:
    if not options:
        return "연금 옵션 정보 없음"

    lines = []
    for opt in options:
        visible = [
            ("pnsn_recp_trm", opt.get("pnsn_recp_trm", "")),
            ("pnsn_recp_trm_nm", opt.get("pnsn_recp_trm_nm", "")),
            ("pnsn_entr_age", opt.get("pnsn_entr_age", "")),
            ("pnsn_entr_age_nm", opt.get("pnsn_entr_age_nm", "")),
            ("mon_paym_atm", opt.get("mon_paym_atm", "")),
            ("mon_paym_atm_nm", opt.get("mon_paym_atm_nm", "")),
            ("paym_prd", opt.get("paym_prd", "")),
            ("paym_prd_nm", opt.get("paym_prd_nm", "")),
            ("pnsn_strt_age", opt.get("pnsn_strt_age", "")),
            ("pnsn_strt_age_nm", opt.get("pnsn_strt_age_nm", "")),
            ("pnsn_recp_amt", opt.get("pnsn_recp_amt", "")),
        ]
        line = "  - " + " / ".join(f"{key}={value}" for key, value in visible if value not in (None, ""))
        lines.append(line)
    return "\n".join(lines)


def _format_options_generic(options: list) -> str:
    if not options:
        return "옵션 정보 없음"

    lines = []
    for opt in options:
        visible = []
        for key, value in opt.items():
            if key in {"fin_co_no", "fin_prdt_cd", "dcls_month"} or value in (None, ""):
                continue
            visible.append(f"{key}={value}")
        if visible:
            lines.append("  - " + " / ".join(visible))
    return "\n".join(lines) if lines else "옵션 정보 없음"


def _format_options(product_type: str, options: list) -> str:
    if product_type == "sinyoung":
        return _format_options_credit(options)
    if product_type in {"dambo_mortgage", "dambo_jeonse"}:
        return _format_options_loan(options)
    if product_type == "deposit":
        return _format_options_deposit(options)
    if product_type == "saving":
        return _format_options_saving(options)
    if product_type == "annuity_saving":
        return _format_options_annuity(options)
    return _format_options_generic(options)


def _merge_products(base_list: list, option_list: list, product_type: str) -> List[Dict]:
    options_by_code: Dict[str, list] = {}
    for opt in option_list:
        key = opt.get("fin_prdt_cd", "") or opt.get("fin_co_no", "")
        options_by_code.setdefault(key, []).append(opt)

    products = []
    for base in base_list:
        product_code = base.get("fin_prdt_cd", "") or base.get("fin_co_no", "")
        options = options_by_code.get(product_code, [])

        products.append({
            "product_type": product_type,
            "bank_name": base.get("kor_co_nm", ""),
            "fin_co_no": base.get("fin_co_no", ""),
            "product_name": base.get("fin_prdt_nm", "") or base.get("kor_co_nm", ""),
            "product_code": product_code,
            "join_way": base.get("join_way", ""),
            "join_member": base.get("join_member", ""),
            "join_deny": base.get("join_deny", ""),
            "special_condition": base.get("spcl_cnd", ""),
            "maturity_interest": base.get("mtrt_int", ""),
            "etc_note": base.get("etc_note", "") or base.get("etc", ""),
            "max_limit": base.get("max_limit", ""),
            "loan_limit": base.get("loan_lmt", ""),
            "extra_cost": base.get("loan_inci_expn", ""),
            "early_repay_fee": base.get("erly_rpay_fee", ""),
            "overdue_rate": base.get("dly_rate", ""),
            "company_homepage": base.get("homp_url", ""),
            "company_tel": base.get("cal_tel", ""),
            "pnsn_kind": base.get("pnsn_kind", ""),
            "pnsn_kind_nm": base.get("pnsn_kind_nm", ""),
            "sale_strt_day": base.get("sale_strt_day", ""),
            "mntn_cnt": base.get("mntn_cnt", ""),
            "prdt_type": base.get("prdt_type", ""),
            "prdt_type_nm": base.get("prdt_type_nm", ""),
            "avg_prft_rate": base.get("avg_prft_rate", ""),
            "dcls_rate": base.get("dcls_rate", ""),
            "guar_rate": base.get("guar_rate", ""),
            "btrm_prft_rate_1": base.get("btrm_prft_rate_1", ""),
            "btrm_prft_rate_2": base.get("btrm_prft_rate_2", ""),
            "btrm_prft_rate_3": base.get("btrm_prft_rate_3", ""),
            "sale_co": base.get("sale_co", ""),
            "fin_co_subm_day": base.get("fin_co_subm_day", ""),
            "rate_info": _format_options(product_type, options),
            "dcls_month": base.get("dcls_month", ""),
            "dcls_strt_day": base.get("dcls_strt_day", ""),
            "dcls_end_day": base.get("dcls_end_day") or "현재",
        })

    return products


def products_to_chunks(products: List[Dict]) -> Tuple[List[str], List[dict]]:
    docs, metadatas = [], []

    for p in products:
        product_type = p["product_type"]
        type_ko = PRODUCT_TYPE_KO.get(product_type, product_type)
        aliases = build_product_aliases(p["product_name"], product_type)

        parts = [
            f"[금융회사명] {p['bank_name']}",
            f"[상품명] {p['product_name']}",
            f"[상품종류] {type_ko}",
        ]

        optional_fields = [
            ("가입방법", p["join_way"]),
            ("가입대상", p["join_member"]),
            ("가입제한", p["join_deny"]),
            ("우대조건", p["special_condition"]),
            ("만기 후 이자율", p["maturity_interest"]),
            ("최고한도", p["max_limit"]),
            ("대출한도", p["loan_limit"]),
            ("부대비용", p["extra_cost"]),
            ("중도상환수수료", p["early_repay_fee"]),
            ("연체이자율", p["overdue_rate"]),
            ("연금종류", _join_present([p["pnsn_kind"], p["pnsn_kind_nm"]])),
            ("판매개시일", p["sale_strt_day"]),
            ("유지건수/설정액", p["mntn_cnt"]),
            ("상품유형", _join_present([p["prdt_type"], p["prdt_type_nm"]])),
            ("평균수익률", p["avg_prft_rate"]),
            ("공시이율", p["dcls_rate"]),
            ("최저보증이율", p["guar_rate"]),
            ("과거수익률1", p["btrm_prft_rate_1"]),
            ("과거수익률2", p["btrm_prft_rate_2"]),
            ("과거수익률3", p["btrm_prft_rate_3"]),
            ("판매사", p["sale_co"]),
            ("금융회사 제출일", p["fin_co_subm_day"]),
            ("기타 유의사항", p["etc_note"]),
            ("회사 홈페이지", p["company_homepage"]),
            ("대표 전화", p["company_tel"]),
        ]
        for label, value in optional_fields:
            if value:
                parts.append(f"[{label}]\n{value}")

        if p["rate_info"]:
            parts.append(f"[옵션 정보]\n{p['rate_info']}")

        if aliases:
            parts.append(f"[검색 별칭]\n{' '.join(aliases)}")

        if p["dcls_month"]:
            parts.append(
                f"[공시기간] {p['dcls_strt_day']} ~ {p['dcls_end_day']} (공시월 {p['dcls_month']})"
            )

        doc_text = "\n".join(parts)
        meta = {
            "bank_name": p["bank_name"],
            "loan_type": product_type,
            "product_type": product_type,
            "product_name": p["product_name"],
            "product_code": p["product_code"],
            "dcls_month": p["dcls_month"],
            "aliases": aliases,
        }

        docs.append(doc_text)
        metadatas.append(meta)

    return docs, metadatas


async def crawl_all() -> Tuple[List[str], List[dict]]:
    if not FSS_API_KEY:
        raise ValueError("FSS_API_KEY 환경변수가 설정되지 않았습니다.")

    all_docs, all_metas = [], []

    for product_type, endpoint in PRODUCT_ENDPOINTS.items():
        print(f"[{PRODUCT_TYPE_KO[product_type]}] FSS API 수집 중...")
        try:
            base_list, option_list = await _fetch_product_pages(product_type, endpoint)
            products = _merge_products(base_list, option_list, product_type)
            docs, metas = products_to_chunks(products)
            all_docs.extend(docs)
            all_metas.extend(metas)
            print(f"  {len(products)}개 상품 -> {len(docs)}개 청크")
        except Exception as e:
            print(f"  [{product_type}] 수집 실패: {e}")

    print(f"\n전체 수집 완료: {len(all_docs)}개 청크")
    return all_docs, all_metas
