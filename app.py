# =========================================================
# KB 금융 RAG 챗봇 (Local Self-Contained Version)
# =========================================================
# 이 코드는 서버나 클라우드 DB 없이, 사용자가 직접 PDF를 업로드하여
# 로컬에서 지식 베이스를 구축하고 질문할 수 있는 구조입니다.
# Groq(LLM), Google(Voice/Translate) API를 사용하여 무료로 동작합니다.
# =========================================================

import os
import sys
import numpy as np
import traceback
import fitz  # PyMuPDF (PDF 처리)
from typing import List

# --- 라이브러리 임포트 ---
import gradio as gr
import speech_recognition as sr
from dotenv import load_dotenv  # 환경 변수 로드 (.env)

# .env 파일 로드 (로컬 개발용)
load_dotenv()

from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # langchain 0.2.0 이상에서 구조가 변경된 경우
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# 1. 설정 및 초기화
# =========================================================

# Groq API 키 (필수)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("⚠️ GROQ_API_KEY가 설정되지 않았습니다. RAG 기능 사용 시 오류가 발생할 수 있습니다.")

# 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
COLLECTION_NAME = "local_kb"

print("🛠️ 모델 및 클라이언트 초기화 중...")

# 1. 임베딩 모델 로드 (로컬 실행)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model.max_seq_length = 512

# 2. Qdrant 클라이언트 (로컬 메모리 DB - 프로그램 종료 시 데이터 삭제됨)
# 영구 저장을 원하면 path="./local_qdrant_db" 로 변경하세요.
# 여기서는 포트폴리오용 데모를 위해 매번 깨끗한 상태인 ':memory:'를 기본으로 합니다.
qdrant_client = QdrantClient(":memory:")

# 컬렉션 생성 (이미 존재하면 삭제 후 재생성)
try:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"✅ 로컬 Qdrant 컬렉션 '{COLLECTION_NAME}' 생성 완료.")
except Exception as e:
    print(f"❌ Qdrant 컬렉션 생성 실패: {e}")

# 3. Groq 클라이언트
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    groq_client = None
    print(f"❌ Groq 클라이언트 초기화 실패: {e}")

#전역 변수: 문서 ID 카운터
doc_id_counter = 0

print("✅ 모든 시스템 준비 완료!")


# =========================================================
# 2. 문서 처리 및 RAG 핵심 로직
# =========================================================

def process_uploaded_files(files):
    """PDF 파일을 읽어 텍스트를 추출하고 벡터 DB에 저장"""
    global doc_id_counter
    
    if not files:
        return "파일이 업로드되지 않았습니다."
    
    total_chunks = 0
    status_msg = ""
    
    # 텍스트 분리기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    for file in files:
        try:
            # Gradio 버전/설정에 따라 file이 문자열(경로)일 수도 있고 객체일 수도 있음
            file_path = file.name if hasattr(file, 'name') else file
            
            # 1. PDF 텍스트 추출
            doc = fitz.open(file_path)
            file_text = ""
            for page in doc:
                file_text += page.get_text()
            
            if not file_text.strip():
                status_msg += f"⚠️ {os.path.basename(file_path)}: 텍스트 추출 실패 (이미지 PDF일 수 있음)\n"
                continue
                
            # 2. 텍스트 분할 (Chunking)
            chunks = text_splitter.split_text(file_text)
            
            # 3. 임베딩 및 저장
            points = []
            for i, chunk in enumerate(chunks):
                vector = embedding_model.encode(chunk).tolist()
                
                payload = {
                    "filename": os.path.basename(file_path),
                    "text": chunk,
                    "chunk_id": i
                }
                
                points.append(PointStruct(id=doc_id_counter, vector=vector, payload=payload))
                doc_id_counter += 1
            
            # Qdrant에 저장
            if points:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                total_chunks += len(points)
                status_msg += f"✅ {os.path.basename(file_path)}: {len(points)}개 지식 저장 완료.\n"
            
        except Exception as e:
            traceback.print_exc()
            file_name_debug = getattr(file, 'name', str(file))
            status_msg += f"❌ {os.path.basename(file_name_debug)} 처리 중 오류: {str(e)}\n"
            
    print(f"DEBUG: 총 저장된 청크 수: {total_chunks}")
    if total_chunks == 0:
        return status_msg + "\n(저장된 데이터가 없습니다. PDF가 비어있거나 이미지일 수 있습니다.)"
            
    return f"처리 완료! 총 {total_chunks}개의 지식 조각이 저장되었습니다.\n\n{status_msg}"

def search_knowledge_base(query, top_k=5):
    """로컬 Qdrant에서 관련 문서 검색"""
    try:
        query_vector = embedding_model.encode(query).tolist()
        # qdrant-client 버전에 따라 .search()가 없거나 다르게 동작할 수 있어 .query_points() 사용
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return search_result.points
    except Exception as e:
        print(f"검색 오류: {e}")
        return []

def generate_answer_groq(query, context_text):
    """Groq API를 사용하여 답변 생성"""
    if not groq_client: 
        return "Groq API 설정 오류"
        
    system_prompt = """
    당신은 친절하고 전문적인 금융 AI 어시스턴트입니다.
    반드시 아래 제공된 [참고자료]만을 바탕으로 질문에 답변하세요.
    참고자료에 내용이 없다면 솔직하게 모른다고 대답하세요.
    출처(파일이름)를 답변 끝에 명시해주세요.
    """
    
    user_prompt = f"질문: {query}\n\n[참고자료]\n{context_text}"
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq 생성 오류: {e}"

# RAG 파이프라인 (통합)
def run_rag_pipeline(text_input, detected_lang='ko'):
    if not text_input:
        return "", "", "", ""
        
    # 1. 질문 번역 (필요시)
    korean_query = text_input
    if detected_lang != 'ko':
        try:
            korean_query = GoogleTranslator(source='auto', target='ko').translate(text_input)
        except: pass

    # 2. 문서 검색
    hits = search_knowledge_base(korean_query)
    
    if not hits:
        return korean_query, "저장된 지식이 부족하여 답변할 수 없습니다. PDF를 먼저 업로드해주세요.", "", "참고 문서 없음"

    # 3. 컨텍스트 구성
    context_text = ""
    references = []
    for hit in hits:
        context_text += f"{hit.payload['text']}\n\n"
        references.append(f"- {hit.payload['filename']} (유사도: {hit.score:.2f})")
    
    ref_str = "\n".join(references)
    
    # 4. 답변 생성
    korean_answer = generate_answer_groq(korean_query, context_text)
    
    # 5. 답변 번역 (필요시)
    final_answer = korean_answer
    if detected_lang != 'ko':
        try:
            final_answer = GoogleTranslator(source='ko', target=detected_lang).translate(korean_answer)
        except: pass
        
    return korean_query, korean_answer, final_answer, ref_str


# =========================================================
# 3. 음성 및 UI 헬퍼 함수
# =========================================================

def voice_to_text(audio_input):
    """음성 인식 (Google API)"""
    if audio_input is None: return "음성 입력 없음", None
    
    try:
        sample_rate, audio_numpy = audio_input
        if audio_numpy.dtype == np.float32:
            audio_numpy = (audio_numpy * 32767).astype(np.int16)
        if len(audio_numpy.shape) > 1:
            audio_numpy = audio_numpy.mean(axis=1).astype(np.int16)
            
        audio_data = sr.AudioData(audio_numpy.tobytes(), sample_rate, 2)
        r = sr.Recognizer()
        text = r.recognize_google(audio_data, language='ko-KR')
        return text, 'ko'
    except sr.UnknownValueError:
        return "인식 실패 (다시 말해주세요)", None
    except Exception as e:
        return f"오류: {e}", None

# =========================================================
# 4. Gradio UI 구성
# =========================================================

# 테마 설정 (KB 금융 색상 - 노란색/회색 톤)
theme = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=theme, title="KB Financial AI Assistant") as demo:
    gr.Markdown(
        """
        # 🏦 KB Financial AI Assistant
        **금융 지식 RAG 시스템**에 오신 것을 환영합니다.
        
        PDF 문서를 업로드하면 AI가 내용을 학습하고, 질문에 대한 정확한 답변과 근거 자료를 제공합니다.
        """
    )
    
    with gr.Accordion("📂 지식 베이스 구축 (Knowledge Base Setup)", open=True):
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="분석할 PDF 문서 업로드 (Drag & Drop)", file_count="multiple", file_types=[".pdf"])
            with gr.Column(scale=1):
                upload_btn = gr.Button("학습 시작 (Build Knowledge Base)", variant="primary")
                upload_status = gr.Textbox(label="시스템 상태", placeholder="대기 중...", interactive=False)
        
    gr.Markdown("---")
    
    with gr.Row():
        # 왼쪽 컬럼: 입력 (음성/텍스트)
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 💬 질문 입력 (Query)")
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="음성으로 질문하기")
            asr_btn = gr.Button("음성 인식 (STT)", variant="secondary")
            
            text_in = gr.Textbox(label="질문 내용", placeholder="궁금한 내용을 입력하세요...", lines=3)
            chat_btn = gr.Button("답변 요청 (Ask AI)", variant="primary", size="lg")
            
        # 오른쪽 컬럼: 결과 (답변/참조)
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### 🤖 분석 결과 (Analysis Result)")
            answer_box = gr.Textbox(label="AI 답변", lines=8, interactive=False, show_copy_button=True)
            ref_box = gr.Textbox(label="참고 문헌 / 근거 자료", lines=4, interactive=False)
            
            
    # 이벤트 연결
    upload_btn.click(process_uploaded_files, inputs=[file_input], outputs=[upload_status])
    
    asr_btn.click(voice_to_text, inputs=[audio_in], outputs=[text_in, gr.State()])
    
    chat_btn.click(
        run_rag_pipeline, 
        inputs=[text_in, gr.State('ko')], # 언어는 기본 한국어로 고정 (단순화)
        outputs=[gr.State(), answer_box, gr.State(), ref_box]
    )

if __name__ == "__main__":
    demo.launch(share=True)
