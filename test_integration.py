"""
전체 Docker 환경 통합 테스트
실행: python test_integration.py

서비스가 모두 기동된 후 실행하세요.
  Qdrant   : http://localhost:6333
  FastAPI  : http://localhost:8000
  Spring   : http://localhost:8080
"""

import httpx
import sys
import time

QDRANT_URL  = "http://localhost:6333"
FASTAPI_URL = "http://localhost:8000"
SPRING_URL  = "http://localhost:8080"

TEST_EMAIL    = "test@test.com"
TEST_PASSWORD = "test1234"
TEST_QUESTION = "국민은행 신용대출 금리 알려줘"

SEP     = "=" * 55
SEP_SUB = "-" * 40
OK  = "✅"
ERR = "❌"
SKP = "⏭️ "

results: list[tuple[str, bool]] = []

def log(name: str, ok: bool, detail: str = ""):
    mark = OK if ok else ERR
    print(f"  {mark} {name}" + (f"  →  {detail}" if detail else ""))
    results.append((name, ok))

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ─────────────────────────────────────────
# 1. Qdrant 헬스체크
# ─────────────────────────────────────────
def check_qdrant():
    section("1. Qdrant 상태 확인")
    try:
        r = httpx.get(f"{QDRANT_URL}/healthz", timeout=5)
        log("Qdrant 응답", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        log("Qdrant 응답", False, str(e))
        return

    try:
        r = httpx.get(f"{QDRANT_URL}/collections", timeout=5)
        cols = r.json().get("result", {}).get("collections", [])
        names = [c["name"] for c in cols]
        log("컬렉션 목록 조회", True, str(names))
    except Exception as e:
        log("컬렉션 목록 조회", False, str(e))
        return

    try:
        r = httpx.get(f"{QDRANT_URL}/aliases", timeout=5)
        aliases = r.json().get("result", {}).get("aliases", [])
        for a in aliases:
            cnt_r = httpx.get(f"{QDRANT_URL}/collections/{a['collection_name']}", timeout=5)
            cnt = cnt_r.json().get("result", {}).get("points_count", "?")
            log(f"alias [{a['alias_name']}]", True, f"→ {a['collection_name']} ({cnt}개 벡터)")
        if not aliases:
            log("alias", False, "없음 — FastAPI 최초 기동 시 벡터화 필요")
    except Exception as e:
        log("alias 조회", False, str(e))


# ─────────────────────────────────────────
# 2. FastAPI 직접 통신
# ─────────────────────────────────────────
def check_fastapi():
    section("2. FastAPI 직접 통신 확인")

    # 헬스체크
    try:
        r = httpx.get(f"{FASTAPI_URL}/docs", timeout=5)
        log("FastAPI /docs", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        log("FastAPI /docs", False, str(e))
        print(f"\n  {SKP} FastAPI가 응답하지 않아 이후 단계를 건너뜁니다.")
        return False

    # /chat 직접 호출 (Spring 거치지 않고)
    print(f"\n  질문: \"{TEST_QUESTION}\"")
    try:
        r = httpx.post(
            f"{FASTAPI_URL}/chat",
            json={"user_id": "test_user", "question": TEST_QUESTION},
            timeout=60,     # BGE-Reranker 첫 추론은 느릴 수 있음
        )
        body = r.json()
        answer = body.get("answer", "")
        log("FastAPI /chat 응답", r.status_code == 200, f"status={r.status_code}")
        if answer:
            print(f"\n  답변 미리보기 (100자):\n  {answer[:100]}...")
        else:
            print(f"  ⚠️  응답 본문: {body}")
    except httpx.TimeoutException:
        log("FastAPI /chat 응답", False, "타임아웃 — BGE 모델 로딩 중일 수 있음, 30초 후 재시도")
    except Exception as e:
        log("FastAPI /chat 응답", False, str(e))

    return True


# ─────────────────────────────────────────
# 3. Spring Boot 통신 (회원가입 → 로그인 → 채팅)
# ─────────────────────────────────────────
def check_spring():
    section("3. Spring Boot 전체 흐름 확인")

    # 헬스체크
    try:
        r = httpx.get(f"{SPRING_URL}/login", timeout=5)
        log("Spring /login 페이지", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        log("Spring /login 접근", False, str(e))
        print(f"\n  {SKP} Spring이 응답하지 않아 이후 단계를 건너뜁니다.")
        return

    # 세션 유지를 위해 httpx.Client 사용
    with httpx.Client(base_url=SPRING_URL, follow_redirects=True, timeout=15) as client:

        # 회원가입
        try:
            r = client.post("/user", data={"email": TEST_EMAIL, "password": TEST_PASSWORD})
            # 이미 가입된 경우 200 또는 redirect 모두 허용
            ok = r.status_code in (200, 302, 400)
            log("회원가입 POST /user", ok, f"status={r.status_code}")
        except Exception as e:
            log("회원가입", False, str(e))

        # 로그인
        try:
            r = client.post(
                "/login",
                data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            # 로그인 성공 → /chat 으로 리다이렉트
            logged_in = r.status_code == 200 and "login?error" not in str(r.url)
            log("로그인 POST /login", logged_in, f"최종 URL={r.url}")
            if not logged_in:
                print(f"  ⚠️  로그인 실패 — 이메일/비밀번호 확인 필요")
                return
        except Exception as e:
            log("로그인", False, str(e))
            return

        # 채팅 API (세션 쿠키 자동 포함)
        try:
            r = client.post(
                "/api/ask",
                json={"question": TEST_QUESTION},
                headers={"Content-Type": "application/json"},
                timeout=90,     # FastAPI → BGE → Gemini 전체 소요
            )
            body = r.json()
            answer = body.get("answer", "")
            log("Spring /api/ask 응답", r.status_code == 200, f"status={r.status_code}")
            if answer:
                print(f"\n  Spring 경유 답변 (100자):\n  {answer[:100]}...")
            else:
                print(f"  ⚠️  응답 본문: {body}")
        except httpx.TimeoutException:
            log("Spring /api/ask", False, "타임아웃 (90초 초과) — FastAPI 로그 확인")
        except Exception as e:
            log("Spring /api/ask", False, str(e))


# ─────────────────────────────────────────
# 4. 결과 요약
# ─────────────────────────────────────────
def summary():
    section("테스트 결과 요약")
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{total} 통과")
    if passed < total:
        print("\n  실패 항목이 있을 경우 아래 로그 명령어로 확인하세요:")
        print("    docker logs fastapi-app  --tail 50")
        print("    docker logs spring-app   --tail 50")
        print("    docker logs finance-mysql --tail 20")


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  Docker 통합 테스트 시작")
    print(f"{'='*55}")

    check_qdrant()
    ok = check_fastapi()
    if ok:
        check_spring()
    summary()
