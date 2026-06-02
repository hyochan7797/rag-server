import os
import asyncio
from dotenv import load_dotenv
from typing import Any, List, Tuple, Optional

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition, Filter, MatchAny,
    CreateAliasOperation, CreateAlias,
    DeleteAliasOperation, DeleteAlias,
)
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

# =======================
# 환경 설정
# =======================
load_dotenv(".env", override=False)
dotenv_path = "/app/.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=False)

api_key         = os.getenv("OPENAI_API_KEY")
qdrant_url      = os.getenv("QDRANT_URL", "http://localhost:6333")
collection_name = os.getenv("COLLECTION_NAME", "loan_docs")
embedding_chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", "8"))
qdrant_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "8"))
embedding_max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "20"))

if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# =======================
# BGE-Reranker 로딩 (로컬)
# =======================
reranker_model: Optional[Any] = None

# =======================
# Qdrant 연결
# =======================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=embedding_chunk_size,
    max_retries=embedding_max_retries,
    retry_min_seconds=1,
    retry_max_seconds=20,
)
client     = QdrantClient(url=qdrant_url)
vectorstore: Optional[Qdrant] = None
_current_backing_collection: Optional[str] = None

# =======================
# BM25 인덱스 (Sparse Search)
# =======================
bm25_index:  Optional[BM25Okapi] = None
bm25_corpus: List[str]  = []
bm25_metas:  List[dict] = []


def _get_reranker_model() -> Any:
    global reranker_model
    if reranker_model is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BGE reranker on first search. device={device}")
        reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device=device)
        print("BGE reranker loaded.")
    return reranker_model


def _normalize_allowed_types(allowed_types: Optional[List[str]]) -> Optional[List[str]]:
    if not allowed_types:
        return allowed_types
    expanded = []
    for loan_type in allowed_types:
        if loan_type == "dambo":
            expanded.extend(["dambo_mortgage", "dambo_jeonse"])
        else:
            expanded.append(loan_type)
    return list(dict.fromkeys(expanded))


def _build_bm25_index(docs: List[str], metas: Optional[List[dict]] = None):
    global bm25_index, bm25_corpus, bm25_metas
    tokenized   = [doc.lower().split() for doc in docs]
    bm25_index  = BM25Okapi(tokenized)
    bm25_corpus = list(docs)
    bm25_metas  = list(metas) if metas else [{} for _ in docs]
    print(f"✅ BM25 인덱스 구축 완료 ({len(docs)}개 문서)")


def _rebuild_bm25_from_qdrant(collection: str):
    """앱 재시작 시 Qdrant 기존 데이터로 BM25 인덱스 재구축"""
    all_docs, all_metas = [], []
    next_offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )
        for point in results:
            text = point.payload.get("page_content", "")
            meta = point.payload.get("metadata", {})
            if text:
                all_docs.append(text)
                all_metas.append(meta)
        if next_offset is None:
            break
    if all_docs:
        _build_bm25_index(all_docs, all_metas)


def _bm25_search(
    query: str,
    k: int = 30,
    allowed_banks: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    if bm25_index is None:
        return []
    allowed_types = _normalize_allowed_types(allowed_types)
    scores  = bm25_index.get_scores(query.lower().split())
    indexed = list(enumerate(scores))

    if allowed_banks or allowed_types:
        indexed = [
            (i, s) for i, s in indexed
            if (not allowed_banks or bm25_metas[i].get("bank_name") in allowed_banks)
            and (not allowed_types or bm25_metas[i].get("loan_type") in allowed_types)
        ]

    top_k = sorted(indexed, key=lambda x: x[1], reverse=True)[:k]
    return [(bm25_corpus[i], float(s)) for i, s in top_k if s > 0]


def _rrf_merge(
    dense_results: List[Tuple[str, float]],
    bm25_results:  List[Tuple[str, float]],
    k: int = 60,
) -> List[str]:
    """Reciprocal Rank Fusion — Dense + BM25 점수 통합"""
    rrf_scores: dict = {}
    for rank, (doc, _) in enumerate(dense_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc, _) in enumerate(bm25_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [doc for doc, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]


# =======================
# Alias 조회
# =======================
def _get_alias_backing() -> Optional[str]:
    try:
        for alias in client.get_aliases().aliases:
            if alias.alias_name == collection_name:
                return alias.collection_name
    except Exception:
        pass
    return None


def init_vectorstore_from_existing() -> bool:
    """Qdrant에 이미 데이터가 있으면 재사용 + BM25 인덱스 복원."""
    global vectorstore, _current_backing_collection
    backing = _get_alias_backing()
    if backing:
        try:
            count = client.count(backing).count
            if count > 0:
                vectorstore = Qdrant(
                    client=client,
                    collection_name=collection_name,
                    embeddings=embeddings,
                )
                _current_backing_collection = backing
                print(f"✅ 기존 컬렉션 재사용: {backing} ({count}개 벡터)")
                print("🔄 BM25 인덱스 재구축 중...")
                _rebuild_bm25_from_qdrant(backing)
                return True
        except Exception as e:
            print(f"⚠️ 기존 컬렉션 확인 실패: {e}")
    print("ℹ️  저장된 벡터 없음 — FSS API로 초기 적재 예정")
    return False


def is_vectorstore_ready() -> bool:
    return vectorstore is not None


# =======================
# Blue/Green 벡터 갱신
# =======================
async def refresh_vectorstore(new_docs: List[str], new_metas: List[dict]) -> bool:
    """
    Qdrant alias 기반 무중단 교체.
    새 backing 컬렉션(v1↔v2)을 만들고 alias를 원자적으로 전환한 뒤 구 컬렉션 삭제.
    BM25 인덱스도 함께 갱신.
    """
    global vectorstore, _current_backing_collection

    old_backing = _current_backing_collection
    new_backing = (
        f"{collection_name}_v2"
        if (old_backing or "").endswith("_v1")
        else f"{collection_name}_v1"
    )

    try:
        print(f"🔄 새 컬렉션 빌드 중: {new_backing} ({len(new_docs)}개 청크)")
        await asyncio.to_thread(
            Qdrant.from_texts,
            new_docs, embeddings,
            metadatas=new_metas,
            url=qdrant_url,
            collection_name=new_backing,
            force_recreate=True,
            batch_size=qdrant_batch_size,
        )

        count = client.count(new_backing).count
        if count < len(new_docs) * 0.9:
            raise ValueError(f"검증 실패: {count}/{len(new_docs)} 적재됨")
        print(f"  ✅ 검증 완료 ({count}개 벡터)")

        # alias 원자적 전환
        alias_operations = []
        if old_backing:
            alias_operations.append(
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=collection_name))
            )
        alias_operations.append(
            CreateAliasOperation(create_alias=CreateAlias(
                collection_name=new_backing, alias_name=collection_name,
            ))
        )
        client.update_collection_aliases(change_aliases_operations=alias_operations)

        if vectorstore is None:
            vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings,
            )

        _current_backing_collection = new_backing
        print(f"  🔁 alias 전환: {old_backing} → {new_backing}")

        if old_backing:
            existing = {c.name for c in client.get_collections().collections}
            if old_backing in existing:
                client.delete_collection(old_backing)
                print(f"  🗑️  구 컬렉션 삭제: {old_backing}")

        # BM25 인덱스 갱신
        _build_bm25_index(new_docs, new_metas)

        print("✅ 벡터 갱신 완료 (Dense + BM25)")
        return True

    except Exception as e:
        print(f"⚠️ 벡터 갱신 실패: {e}")
        existing = {c.name for c in client.get_collections().collections}
        if new_backing in existing:
            client.delete_collection(new_backing)
        return False


# =======================
# 리랭킹
# =======================
def _rerank_local(query: str, docs: List[str], top_k: int = 3) -> Tuple[List[str], List[float]]:
    if not docs:
        return [], []
    model_inputs = [[query, doc] for doc in docs]
    scores  = _get_reranker_model().predict(model_inputs)
    results = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results[:top_k]], [float(s) for s, _ in results[:top_k]]


# =======================
# 하이브리드 검색 (Dense + BM25 + BGE Reranker)
# =======================
async def search_similar_docs(
    history_list: list,
    query: str,
    allowed_banks: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None,
    rewritten_query: Optional[str] = None,
    hyde_query: Optional[str] = None,
    top_k: int = 3,
    candidate_k: int = 30,
) -> Tuple[List[str], List[float]]:

    if vectorstore is None:
        print("⚠️ 벡터스토어 미준비 상태 — /admin/refresh 로 데이터를 먼저 적재하세요.")
        return [], []

    if rewritten_query:
        # 쿼리 재작성이 이미 대화 맥락을 반영했으므로 이중 확장 방지
        full_query = rewritten_query
    else:
        recent_context = " ".join(
            msg["content"] for msg in history_list[-2:] if msg["role"] == "user"
        )
        full_query = (recent_context + " " + query).strip()

    allowed_types = _normalize_allowed_types(allowed_types)
    filters = []
    if allowed_banks:
        print(f"🔎 [필터] 은행: {allowed_banks}")
        filters.append(FieldCondition(key="metadata.bank_name", match=MatchAny(any=allowed_banks)))
    if allowed_types:
        print(f"🔎 [필터] 종류: {allowed_types}")
        filters.append(FieldCondition(key="metadata.loan_type", match=MatchAny(any=allowed_types)))

    qdrant_filter = Filter(must=filters) if filters else None
    dense_query = hyde_query or full_query
    if hyde_query:
        print("HyDE Dense query enabled")

    # 1단계: Dense 검색 (의미 기반)
    try:
        search_results = vectorstore.similarity_search_with_score(
            dense_query, k=candidate_k, filter=qdrant_filter
        )
        print(f"🔵 Dense 검색: {len(search_results)}개")
    except Exception as e:
        print(f"⚠️ Qdrant 검색 오류: {e}")
        return [], []

    if not search_results:
        return [], []

    dense_results = [(doc.page_content, float(score)) for doc, score in search_results]

    # 2단계: BM25 검색 (키워드 기반)
    bm25_results = _bm25_search(
        full_query, k=candidate_k,
        allowed_banks=allowed_banks,
        allowed_types=allowed_types,
    )
    print(f"🟡 BM25 검색: {len(bm25_results)}개")

    # 3단계: RRF로 두 결과 통합
    if bm25_results:
        candidate_docs = _rrf_merge(dense_results, bm25_results)[:candidate_k]
        print(f"🔀 RRF 통합: {len(candidate_docs)}개 후보")
    else:
        candidate_docs = [doc for doc, _ in dense_results]
        print(f"📊 Dense only (BM25 결과 없음): {len(candidate_docs)}개")

    # 4단계: BGE-Reranker 최종 정렬
    print("⚡ BGE 리랭킹 중...")
    final_docs, final_scores = await asyncio.to_thread(
        _rerank_local, full_query, candidate_docs, top_k
    )

    return final_docs, final_scores
