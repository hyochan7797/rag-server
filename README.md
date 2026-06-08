# RAG Server — 금융 상품 AI 상담 서버

금융감독원(FSS) 공개 API에서 대출·예금·적금·연금저축 데이터를 수집하고,  
**Hybrid RAG**(Dense + BM25 + BGE Reranker)로 검색해 Gemini가 답변하는 FastAPI 서버입니다.

---

## 아키텍처

```
사용자 질문
   │
   ├─ 필터 추출 (은행·상품 종류)      filter_extraction.py
   ├─ 쿼리 재작성 (Gemini)            main.py
   ├─ 도메인 동의어 확장              query_expansion.py
   │
   ├─ Dense 검색  → Qdrant (OpenAI Embeddings)
   ├─ Sparse 검색 → BM25 (rank-bm25)
   ├─ RRF 통합    → Reciprocal Rank Fusion
   ├─ Reranking   → BGE-Reranker-v2-m3 (로컬 CrossEncoder)
   │
   └─ 답변 생성 → Gemini 2.5 Flash
```

**Blue/Green 무중단 갱신** — `/admin/refresh` 호출 시 새 Qdrant 컬렉션을 빌드한 뒤  
alias를 원자적으로 전환해 서비스 중단 없이 벡터 DB를 교체합니다.

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| API 서버 | FastAPI + Uvicorn |
| 벡터 DB | Qdrant |
| 임베딩 | OpenAI `text-embedding-3-small` |
| Sparse 검색 | BM25Okapi (rank-bm25) |
| Reranker | `BAAI/bge-reranker-v2-m3` (sentence-transformers) |
| 생성 모델 | Google Gemini 2.5 Flash |
| 데이터 수집 | 금융감독원 금융상품통합비교공시 API |
| 컨테이너 | Docker Compose |

---

## 지원 상품 종류

- `sinyoung` — 개인신용대출
- `dambo_mortgage` — 주택담보대출
- `dambo_jeonse` — 전세자금대출
- `deposit` — 정기예금
- `saving` — 적금
- `annuity_saving` — 연금저축
- `company` — 금융회사 정보

---

## 빠른 시작

### 1. 환경 변수 설정

```bash
cp .env.example .env
```

`.env`에 실제 키 입력:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
FSS_API_KEY=...
ADMIN_API_KEY=변경할-긴-랜덤-키   # Spring 서버의 ADMIN_API_KEY와 동일해야 함
```

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | 임베딩 생성용 (text-embedding-3-small) |
| `GOOGLE_API_KEY` | Gemini 답변 생성 + 쿼리 재작성 |
| `FSS_API_KEY` | 금융감독원 공시 API |
| `ADMIN_API_KEY` | `/admin/refresh` 엔드포인트 인증 키 |
| `QDRANT_URL` | Qdrant 주소 (기본값: `http://localhost:6333`) |
| `COLLECTION_NAME` | 컬렉션 이름 (기본값: `loan_docs`) |
| `EMBEDDING_CHUNK_SIZE` | 임베딩 배치 크기 (기본값: `8`) |
| `QDRANT_BATCH_SIZE` | Qdrant 업로드 배치 크기 (기본값: `8`) |
| `EMBEDDING_MAX_RETRIES` | 임베딩 API 재시도 횟수 (기본값: `20`) |

### 2. Docker로 실행 (권장)

```bash
docker compose up -d --build
```

### 3. Python 로컬 실행

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 벡터 데이터 초기 적재

Qdrant가 올라온 뒤 최초 한 번 실행합니다.  
FSS API에서 데이터를 수집해 임베딩 후 Qdrant에 저장합니다.

```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

---

## API 엔드포인트

### `GET /health`

서버 및 벡터스토어 상태 확인

```json
{ "status": "ok", "vectorstore_ready": true }
```

---

### `POST /chat`

금융 상품 질의응답

**Request**

```json
{
  "user_id": "user123",
  "question": "전세대출 금리 낮은 곳 추천해줘",
  "history": null,
  "history_summary": null,
  "use_hyde": false
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `user_id` | string \| int | 사용자 식별자 (대화 히스토리 관리) |
| `question` | string | 사용자 질문 |
| `history` | array \| null | 직접 전달할 대화 이력 (없으면 서버 내부 저장 사용) |
| `history_summary` | string \| null | 장기 대화 요약 (선택) |
| `use_hyde` | bool | HyDE(가상 문서 생성) 사용 여부 (기본 `false`) |

**Response**

```json
{
  "answer": "국민은행 전세자금대출의 금리는 ...\n\nAI 신뢰도: 높음"
}
```

응답 끝에 AI 신뢰도(`매우 높음` / `높음` / `보통`)가 표시됩니다.

**예시**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "question": "국민은행 신용대출 한도 알려줘"}'
```

---

### `POST /admin/refresh`

FSS API 재수집 + 벡터 DB 무중단 갱신 (Blue/Green)

```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

**Response**

```json
{ "status": "ok", "chunks": 1234 }
```

---

## Spring 서버(EC2)와 연동

로컬 FastAPI를 ngrok으로 외부 공개:

```bash
ngrok http 8000
```

Spring EC2의 `.env`에 설정:

```env
FASTAPI_URL=https://YOUR_NGROK_DOMAIN/chat
FASTAPI_ADMIN_URL=https://YOUR_NGROK_DOMAIN/admin/refresh
ADMIN_API_KEY=fastapi와_동일한_값
LOAN_REFRESH_CRON=-
```

Spring 재시작:

```bash
docker compose -f docker-compose.spring-aws.yml pull
docker compose -f docker-compose.spring-aws.yml up -d
```

---

## 주요 동작 흐름

1. **데이터 수집** — `fss_crawler.py`가 금융감독원 API에서 상품 데이터 병렬 수집
2. **벡터 적재** — OpenAI 임베딩 → Qdrant, BM25 인덱스 동시 구축
3. **쿼리 처리**
   - 필터 추출 (은행명, 상품 종류 정규식 매칭)
   - Gemini로 쿼리 재작성 (대화 맥락 반영)
   - 도메인 동의어 확장 (주담대 → 주택담보대출 등)
4. **하이브리드 검색**
   - Dense: Qdrant 의미 검색 (Cosine)
   - Sparse: BM25 키워드 검색
   - RRF로 두 결과 통합
5. **Reranking** — BGE-Reranker-v2-m3로 최종 Top-K 선별
6. **답변 생성** — 참고 문서를 컨텍스트로 Gemini 2.5 Flash가 답변

---

## 참고

- 서버 시작 시 Qdrant에 기존 데이터가 있으면 자동 재사용 (BM25 인덱스도 복원)
- 데이터가 없는 상태에서 채팅 요청 시 갱신 안내 메시지 반환
- 대화 히스토리는 유저당 최근 10개 메시지 인메모리 유지
