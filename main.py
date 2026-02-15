from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import traceback
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="KB AI RAG Service")

# CORS 해결 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 설정 ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
COLLECTION_NAME = "local_kb"

# --- 전역 변수 (초기화) ---
embedding_model = None
qdrant_client = None
groq_client = None
doc_id_counter = 0

@app.on_event("startup")
async def startup_event():
    global embedding_model, qdrant_client, groq_client
    print("🚀 서버 시작: 모델 로딩 중...")
    
    # 1. 임베딩 모델
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 2. Qdrant (메모리 모드)
    qdrant_client = QdrantClient(":memory:")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    # 3. Groq
    if GROQ_API_KEY and GROQ_API_KEY != "gsk_...":
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        print("⚠️ Groq API 키가 설정되지 않았습니다.")

    print("✅ 서버 준비 완료!")

# --- API 정의 ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "KB AI RAG Service is running"}

@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global doc_id_counter
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    saved_chunks = 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    results = []
    
    for file in files:
        try:
            # 임시 파일 저장 및 읽기
            temp_filename = f"temp_{file.filename}"
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # PDF 텍스트 추출
            doc = fitz.open(temp_filename)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            os.remove(temp_filename) # 정리
            
            if not text.strip():
                results.append({"filename": file.filename, "status": "failed", "reason": "No text extracted"})
                continue

            # 청크화 및 임베딩
            chunks = text_splitter.split_text(text)
            points = []
            for i, chunk in enumerate(chunks):
                vector = embedding_model.encode(chunk).tolist()
                points.append(PointStruct(
                    id=doc_id_counter,
                    vector=vector,
                    payload={"filename": file.filename, "text": chunk}
                ))
                doc_id_counter += 1
            
            if points:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                saved_chunks += len(points)
                results.append({"filename": file.filename, "status": "success", "chunks": len(points)})
                
        except Exception as e:
            traceback.print_exc()
            results.append({"filename": file.filename, "status": "error", "message": str(e)})

    return {"total_chunks": saved_chunks, "details": results}

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not groq_client:
        return {"answer": "죄송합니다. 서버 설정 오류(API Key 누락)입니다.", "references": []}
    
    try:
        # 1. 검색
        query_vector = embedding_model.encode(request.query).tolist()
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )
        
        if not hits:
            return {"answer": "관련 정보를 찾을 수 없습니다. 문서를 먼저 업로드해주세요.", "references": []}

        # 2. 컨텍스트 구성
        context = "\n\n".join([h.payload['text'] for h in hits])
        refs = [h.payload['filename'] for h in hits]
        
        # 3. LLM 생성 (Groq)
        system_prompt = "당신은 금융 AI 어시스턴트입니다. 주어진 [참고자료]를 바탕으로 질문에 답변하세요. 출처를 꼭 명시하세요."
        user_prompt = f"질문: {request.query}\n\n[참고자료]\n{context}"
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.1
        )
        
        return {"answer": response.choices[0].message.content, "references": list(set(refs))}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
