# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# 필수 시스템 패키지 설치 (빌드 실패 방지)
RUN apt-get update && apt-get install -y \
    gcc g++ make build-essential libssl-dev libffi-dev python3-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN pip install --upgrade pip setuptools wheel

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 전체 복사
COPY . .

# 포트 개방
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]