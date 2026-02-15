# =========================================================
# KB AI Challenge - Professional RAG System (Multilingual)
# =========================================================

import os
import sys
import numpy as np
import traceback
import fitz  # PyMuPDF
from typing import List

# --- 라이브러리 임포트 ---
import gradio as gr
import speech_recognition as sr
from dotenv import load_dotenv

# .env 로드
load_dotenv()

from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# 1. 설정 및 초기화
# =========================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
COLLECTION_NAME = "local_kb"

print("🛠️ 시스템 초기화 중... (System Init)")

# 모델 로드
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model.max_seq_length = 512

# Qdrant (메모리)
qdrant_client = QdrantClient(":memory:")
try:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"✅ Qdrant Collection Ready.")
except Exception as e:
    print(f"❌ Qdrant Error: {e}")

# Groq Init
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"❌ Groq Error: {e}")
else:
    print("⚠️ Groq API Key Missing.")

doc_id_counter = 0

print("✅ System Ready.")


# =========================================================
# 2. 다국어 지원 로직 (Translation & STT)
# =========================================================

LANG_MAP = {
    "한국어 (Korean)": {"code": "ko", "stt": "ko-KR"},
    "English (영어)": {"code": "en", "stt": "en-US"},
    "日本語 (Japanese)": {"code": "ja", "stt": "ja-JP"},
    "中文 (Chinese)": {"code": "zh-CN", "stt": "zh-CN"}
}

def translate_text(text, target_lang_code):
    try:
        if target_lang_code == "ko": return text
        return GoogleTranslator(source='auto', target=target_lang_code).translate(text)
    except:
        return text

def translate_to_korean(text):
    try:
        return GoogleTranslator(source='auto', target='ko').translate(text)
    except:
        return text

# =========================================================
# 3. 핵심 로직 (RAG Pipeline)
# =========================================================

def process_uploaded_files(files):
    """PDF 처리 및 임베딩"""
    global doc_id_counter
    if not files: return "파일이 선택되지 않았습니다."
    
    total_chunks = 0
    status_msg = ""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)

    for file in files:
        try:
            file_path = file.name if hasattr(file, 'name') else file
            doc = fitz.open(file_path)
            file_text = ""
            for page in doc: file_text += page.get_text()
            
            if not file_text.strip():
                status_msg += f"⚠️ {os.path.basename(file_path)}: 텍스트 없음.\n"
                continue
                
            chunks = text_splitter.split_text(file_text)
            points = []
            for i, chunk in enumerate(chunks):
                vector = embedding_model.encode(chunk).tolist()
                payload = {"filename": os.path.basename(file_path), "text": chunk}
                points.append(PointStruct(id=doc_id_counter, vector=vector, payload=payload))
                doc_id_counter += 1
            
            if points:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_chunks += len(points)
                status_msg += f"✅ {os.path.basename(file_path)} ({len(points)} 개 저장됨)\n"
            
        except Exception as e:
            status_msg += f"❌ 오류: {os.path.basename(file_path)} - {str(e)}\n"
            
    return f"총 {total_chunks}개 데이터 처리 완료.\n\n{status_msg}"

def search_knowledge_base(query, top_k=5):
    try:
        query_vector = embedding_model.encode(query).tolist()
        res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=top_k, with_payload=True
        )
        return res.points
    except:
        return []

def generate_answer_groq(query, context_text):
    if not groq_client: return "API 키가 필요합니다."
    
    system_prompt = """
    당신은 KB 금융그룹의 전문 AI 어시스턴트입니다.
    제공된 [문맥]에 기반하여 질문에 대해 정확하고 전문적인 답변을 작성하세요.
    모르는 내용은 모른다고 답하고, 추측하지 마세요.
    답변은 한국어로 작성하세요.
    """
    user_prompt = f"질문: {query}\n\n[문맥]\n{context_text}"
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=GROQ_MODEL_NAME, temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"응답 생성 오류: {e}"

def run_rag_chat(message, history, lang_selection):
    if not message: return "", history, ""
    
    target_lang = LANG_MAP[lang_selection]["code"]
    
    # 1. 입력 번역 (Target -> Korean)
    korean_query = message
    if target_lang != "ko":
        korean_query = translate_to_korean(message)
    
    # 2. 검색 & 답변 생성 (Korean)
    hits = search_knowledge_base(korean_query)
    if not hits:
        bot_response_ko = "죄송합니다. 관련 정보를 찾을 수 없습니다."
        reference_text = "참고 문서 없음"
    else:
        context_text = "\n\n".join([h.payload['text'] for h in hits])
        # 중복 제거 및 그룹화 (File grouping)
        ref_data = {}
        for h in hits:
            fname = h.payload['filename']
            if fname not in ref_data:
                ref_data[fname] = []
            ref_data[fname].append(h.score)
            
        refs = []
        for fname, scores in ref_data.items():
            refs.append(f"- {fname} (관련 내용 {len(scores)}건, 최고 유사도: {max(scores):.2f})")
        reference_text = "\n".join(refs)
        bot_response_ko = generate_answer_groq(korean_query, context_text)
    
    # 3. 답변 번역 (Korean -> Target)
    final_response = bot_response_ko
    if target_lang != "ko":
        translated_response = translate_text(bot_response_ko, target_lang)
        final_response = f"{translated_response}\n\n---\n[한국어 원문]\n{bot_response_ko}"
    
    # 히스토리에 추가 (Messages Format for Gradio 6.x)
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_response}
    ]
    return "", new_history, reference_text

def voice_to_text_chat(audio, history, lang_selection):
    if audio is None: return "", history, "음성 입력 없음"
    
    stt_lang = LANG_MAP[lang_selection]["stt"]
    
    try:
        sample_rate, audio_numpy = audio
        if audio_numpy.dtype == np.float32:
            audio_numpy = (audio_numpy * 32767).astype(np.int16)
        if len(audio_numpy.shape) > 1:
            audio_numpy = audio_numpy.mean(axis=1).astype(np.int16)
        audio_data = sr.AudioData(audio_numpy.tobytes(), sample_rate, 2)
        r = sr.Recognizer()
        
        # 선택된 언어로 인식
        text = r.recognize_google(audio_data, language=stt_lang)
        
        # 채팅 함수 호출
        return run_rag_chat(text, history, lang_selection)
        
    except sr.UnknownValueError:
        return "", history, "음성을 이해할 수 없습니다."
    except Exception as e: 
        return "", history, f"오류: {e}"

# =========================================================
# 4. UI Layout (Clean Professional Korean)
# =========================================================

theme = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "sans-serif"]
)

css = """
footer {visibility: hidden !important;}
.gradio-container {min-height: 0px !important;}
"""

with gr.Blocks(theme=theme, title="KB AI Challenge", css=css) as demo:
    
    with gr.Row():
        # --- LEFT SIDEBAR ---
        with gr.Column(scale=1, min_width=300, variant="panel"):
            gr.Markdown("## KB AI Challenge")
            gr.Markdown("**다국어 금융 AI 어시스턴트**")
            
            with gr.Group():
                lang_dropdown = gr.Dropdown(
                    choices=list(LANG_MAP.keys()), 
                    value="한국어 (Korean)", 
                    label="언어 설정",
                    interactive=True
                )
                
                file_input = gr.File(label="지식 베이스 (PDF)", file_count="multiple", file_types=[".pdf"])
                with gr.Row():
                    upload_btn = gr.Button("업로드 및 분석", variant="primary", size="sm")
                    upload_status = gr.Textbox(show_label=False, placeholder="상태 대기 중...", interactive=False, lines=1, max_lines=1)
            
            gr.Markdown("### 음성 대화")
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="음성 입력", show_label=False)
            
            with gr.Accordion("시스템 아키텍처", open=False):
                gr.Markdown(
                    """
                    **최적화 내역**
                    1. **STT**: Google Speech API
                    2. **번역**: Google Translate API
                    3. **LLM**: Groq LPU (Llama 3)
                    """
                )

        # --- RIGHT MAIN ---
        with gr.Column(scale=3):
            # chatbot (Messages format)
            chatbot = gr.Chatbot(label="대화", height=500, show_label=False)
            
            # References
            gr.Markdown("**참고 문서**")
            ref_output = gr.Textbox(show_label=False, interactive=False, lines=3, max_lines=5, placeholder="관련 문서가 표시됩니다.")
            
            # Input Area
            with gr.Row():
                msg = gr.Textbox(
                    scale=6, 
                    show_label=False, 
                    placeholder="질문을 입력하세요...",
                    container=False
                )
                submit_btn = gr.Button("전송", scale=1, variant="primary")

    # --- Event Handlers ---
    upload_btn.click(process_uploaded_files, inputs=[file_input], outputs=[upload_status])
    
    msg.submit(run_rag_chat, [msg, chatbot, lang_dropdown], [msg, chatbot, ref_output])
    submit_btn.click(run_rag_chat, [msg, chatbot, lang_dropdown], [msg, chatbot, ref_output])
    
    audio_input.stop_recording(voice_to_text_chat, [audio_input, chatbot, lang_dropdown], [msg, chatbot, ref_output])

if __name__ == "__main__":
    demo.launch(share=True)
