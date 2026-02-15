# KB AI Challenge

이 프로젝트는 KB 금융 관련 문서를 업로드하고 질문할 수 있는 **RAG(검색 증강 생성) API 서비스**입니다.
FastAPI를 사용하여 구축되었으며, 로컬 Qdrant와 무료 Groq(Llama 3) API를 활용합니다.

## ✨ 주요 기능
- **PDF 업로드 및 자동 인덱싱**: `/upload` API를 통해 문서를 즉시 벡터 DB에 저장합니다.
- **실시간 질의응답**: `/chat` API를 통해 저장된 지식을 바탕으로 답변합니다.
- **완전 무료 구조**: 서버리스 배포 가능, 무료 LLM 활용.

## 🛠️ 기술 스택
- **Backend**: FastAPI
- **LLM**: Groq (Llama-3.3-70b-versatile)
- **Vector DB**: Qdrant (Local In-Memory)
- **Embedding**: SentenceTransformers (`ko-sroberta-multitask`)

## 🚀 실행 방법

### 1. 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (필수)
실행 전 Groq API 키를 환경 변수로 설정해야 합니다.
```bash
export GROQ_API_KEY="your_actual_groq_api_key"
```
(또는 코드 내 `main.py`의 `GROQ_API_KEY` 변수에 직접 입력하여 로컬 테스트 가능 - **GitHub 업로드 시 주의!**)

### 3. 실행
```bash
uvicorn main:app --reload
```
서버가 실행되면 `http://127.0.0.1:8000/docs` 에서 API를 테스트할 수 있습니다.

## 🐳 Docker 배포
```bash
docker build -t kb-rag-service .
docker run -p 8000:8000 kb-rag-service
```
