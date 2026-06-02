import asyncio
import os
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from filter_extraction import extract_filters_from_query
from fss_crawler import crawl_all
from query_expansion import expand_domain_synonyms
from rag_pipeline import (
    init_vectorstore_from_existing,
    is_vectorstore_ready,
    refresh_vectorstore,
    search_similar_docs,
)

app = FastAPI()
chat_histories: Dict[str, List[Dict[str, str]]] = {}
generation_model = None

load_dotenv(".env", override=False)
load_dotenv("/app/.env", override=False)


def configure_gemini():
    global generation_model
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("[main.py] GOOGLE_API_KEY is not set.")
        generation_model = None
        return

    try:
        genai.configure(api_key=google_api_key)
        generation_model = genai.GenerativeModel("gemini-2.5-flash")
        print("[main.py] Gemini configured.")
    except Exception as e:
        print(f"[main.py] Gemini configuration failed: {e}")
        generation_model = None


configure_gemini()


class ChatRequest(BaseModel):
    user_id: Union[str, int]
    question: str
    history: Optional[List[Dict[str, str]]] = None
    use_hyde: bool = False


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vectorstore_ready": is_vectorstore_ready(),
    }


async def rewrite_query(question: str, history: List[Dict[str, str]]) -> str:
    if generation_model is None:
        return question

    prev_user_msgs = [m["content"] for m in history if m["role"] == "user"]
    context_str = " / ".join(prev_user_msgs[-4:-1]) if len(prev_user_msgs) > 1 else "none"
    prompt = f"""
You are optimizing a Korean finance RAG search query.
Use the previous conversation only to resolve missing context.
Return only one concise Korean search query.

Previous user context: {context_str}
Current question: {question}
"""
    try:
        resp = await asyncio.to_thread(generation_model.generate_content, prompt)
        rewritten = resp.text.strip().strip('"').strip("'")
        return rewritten or question
    except Exception as e:
        print(f"Query rewrite failed; using original question: {e}")
        return question


async def generate_hyde_query(question: str, search_query: str) -> Optional[str]:
    if generation_model is None:
        return None

    prompt = f"""
Write a short hypothetical Korean finance product document for dense retrieval.
Do not invent exact numbers. Use terms like rate, limit, eligibility, and conditions.
Return only the document text.

Original question: {question}
Search query: {search_query}
"""
    try:
        resp = await asyncio.to_thread(generation_model.generate_content, prompt)
        hyde = resp.text.strip().strip('"').strip("'")
        return hyde or None
    except Exception as e:
        print(f"HyDE generation failed: {e}")
        return None


@app.post("/chat")
async def ask_chat(query: ChatRequest):
    user_id = str(query.user_id)
    question = query.question

    if query.history is not None:
        history = list(query.history) + [{"role": "user", "content": question}]
    else:
        history = chat_histories.setdefault(user_id, [])
        history.append({"role": "user", "content": question})

    if not is_vectorstore_ready():
        return {"answer": "데이터가 아직 적재되지 않았습니다. 관리자에게 데이터 갱신을 요청해주세요."}

    final_banks, final_types = extract_filters_from_query(question)
    rewritten = await rewrite_query(question, history)
    expanded_query = expand_domain_synonyms(rewritten)
    hyde_query = await generate_hyde_query(question, expanded_query) if query.use_hyde else None

    top_docs, top_scores = await search_similar_docs(
        history_list=history,
        query=question,
        allowed_banks=final_banks,
        allowed_types=final_types,
        rewritten_query=expanded_query,
        hyde_query=hyde_query,
    )

    if not top_docs:
        return {"answer": "죄송합니다. 해당 조건에 맞는 금융 상품 정보를 찾을 수 없습니다."}

    if top_scores and top_scores[0] < -2.0:
        return {"answer": "죄송합니다. 질문과 관련된 정보를 문서에서 찾을 수 없습니다."}

    context_str = ""
    for i, doc in enumerate(top_docs):
        context_str += f"\n[참고 문서 {i + 1}]\n{doc}\n"

    system_prompt = f"""
당신은 금융 상품 상담 AI입니다.
반드시 아래 참고 문서에 있는 내용만 바탕으로 정확하고 친절하게 답변하세요.
문서에 없는 내용은 확인할 수 없다고 말하세요.

[참고 문서]
{context_str}
"""

    if generation_model is None:
        return {"answer": "답변 생성 모델이 설정되지 않았습니다."}

    try:
        resp = await asyncio.to_thread(
            generation_model.generate_content,
            [{"role": "user", "parts": [system_prompt + "\n\n고객 질문: " + question]}],
        )
        final_answer = resp.text
        best_score = top_scores[0] if top_scores else 0
        if best_score > 0.5:
            reliability_msg = "\n\nAI 신뢰도: 매우 높음"
        elif best_score > -1.0:
            reliability_msg = "\n\nAI 신뢰도: 높음"
        else:
            reliability_msg = "\n\nAI 신뢰도: 보통"

        final_response = final_answer + reliability_msg
        if query.history is None:
            history.append({"role": "model", "content": final_response})
        return {"answer": final_response}
    except Exception as e:
        print(f"Gemini answer generation failed: {e}")
        return {"answer": "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."}


@app.post("/admin/refresh")
async def refresh_vectors(x_admin_key: str = Header(...)):
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="인증 실패")

    print("[admin/refresh] FSS crawl started.")
    try:
        new_docs, new_metas = await crawl_all()
        if not new_docs:
            return {"status": "error", "message": "수집된 데이터가 없습니다."}

        success = await refresh_vectorstore(new_docs, new_metas)
        if success:
            return {"status": "ok", "chunks": len(new_docs)}
        return {"status": "error", "message": "벡터 갱신 실패. 로그를 확인하세요."}
    except Exception as e:
        print(f"[admin/refresh] failed: {e}")
        return {"status": "error", "message": str(e)}


@app.on_event("startup")
async def startup_event():
    if init_vectorstore_from_existing():
        print("[startup] Existing Qdrant data loaded.")
    else:
        print("[startup] Qdrant has no usable data. Run /admin/refresh before chat queries.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
