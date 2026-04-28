"""
LexRAG Backend v3.1 — Full Assignment 3 HCMUT Edition
Hệ thống RAG nâng cao kết hợp kết quả từ Assignment 1 & 2
"""
import os
import json
import time
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# -- Cấu hình Logging --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -- Cấu hình Hệ thống --
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_BASE  = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_K    = 4

app = FastAPI(
    title="LexRAG - Advanced Contract Intelligence",
    description="Hệ thống RAG sử dụng tri thức từ Assignment 1 & 2 để phân tích hợp đồng.",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Data Models --
class NERItem(BaseModel):
    text: str
    label: str
    description: Optional[str] = ""

class SRLFrame(BaseModel):
    predicate: str
    roles: Dict[str, str]
    explanation: str

class ClauseMetadata(BaseModel):
    clause_text: str
    intent: str
    ner: List[NERItem]
    srl: List[SRLFrame]
    score: Optional[float] = 0.0

class QueryRequest(BaseModel):
    query: str = Field(..., example="Khi nào Bên B bị phạt?")
    k: int = Field(default=DEFAULT_K, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: List[ClauseMetadata]
    pipeline_steps: List[str]
    processing_time: float

# -- Vector Store Engine --
class LegalVectorStore:
    def __init__(self):
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dim: int = 0

    def load_resources(self):
        if self.embedder is None:
            logger.info(f"Đang tải mô hình embedding: {EMBED_MODEL}...")
            self.embedder = SentenceTransformer(EMBED_MODEL)
            self.dim = self.embedder.get_sentence_embedding_dimension()
            logger.info("Tải mô hình thành công.")

    def add_documents(self, data_list: List[Dict[str, Any]]):
        self.load_resources()
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        
        # Trích xuất văn bản từ kết quả Ass 2 để tạo vector
        texts = [item["clause"] for item in data_list]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        self.index.add(np.array(embeddings, dtype="float32"))

        # Lưu trữ dữ liệu cấu trúc chi tiết
        for item in data_list:
            self.metadata.append({
                "clause_text": item["clause"],
                "intent": item.get("intent", "Unknown"),
                "ner": item.get("ner", []),
                "srl": item.get("srl", [])
            })
        return len(texts)

    def search(self, query: str, k: int):
        if not self.metadata or self.index is None:
            return []
        
        self.load_resources()
        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_vec, dtype="float32"), min(k, len(self.metadata)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                meta = self.metadata[idx].copy()
                meta["score"] = float(score)
                results.append(meta)
        return results

    def reset(self):
        self.index = None
        self.metadata = []

vs = LegalVectorStore()

# -- Endpoints --

@app.post("/upload", tags=["Data Management"])
async def upload_ass2_data(file: UploadFile = File(...)):
    """Nạp file combined_results.json từ Assignment 2"""
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError("Định dạng file JSON phải là danh sách các mệnh đề.")
        
        count = vs.add_documents(data)
        logger.info(f"Đã nạp {count} mệnh đề từ file {file.filename}")
        return {"message": f"Thành công! Đã nạp {count} mệnh đề giàu ngữ cảnh từ Ass 2.", "count": count}
    except Exception as e:
        logger.error(f"Lỗi nạp file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý file: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def handle_query(req: QueryRequest):
    """Xử lý truy vấn RAG kết hợp Metadata từ Ass 2"""
    start_time = time.time()
    steps = []

    # 1. Retrieval
    steps.append("embedding_query")
    sources = vs.search(req.query, k=req.k)
    steps.append(f"faiss_retrieve_top_{len(sources)}")

    if not sources:
        return QueryResponse(
            answer="Hệ thống chưa có dữ liệu hợp đồng. Vui lòng tải lên file JSON từ Ass 2.",
            sources=[],
            pipeline_steps=steps,
            processing_time=time.time() - start_time
        )

    # 2. Xây dựng Context giàu tri thức (NER + SRL + Intent)
    context_blocks = []
    for i, s in enumerate(sources):
        ner_text = ", ".join([f"{e['text']} ({e['label']})" for e in s['ner']])
        srl_text = s['srl'][0]['explanation'] if s['srl'] else "Không có phân tích vai trò."
        
        block = (
            f"--- [Mệnh đề {i+1}] ---\n"
            f"Nội dung: {s['clause_text']}\n"
            f"Ý định: {s['intent']}\n"
            f"Thực thể: {ner_text}\n"
            f"Phân tích hành vi: {srl_text}"
        )
        context_blocks.append(block)
    
    full_context = "\n\n".join(context_blocks)

    # 3. Prompt Engineering (Tuân thủ yêu cầu 3.3.3 không ảo giác)
    system_prompt = f"""Bạn là một Luật sư AI tư vấn hợp đồng chuyên nghiệp.
Nhiệm vụ: Trả lời câu hỏi của người dùng dựa trên Ngữ cảnh pháp lý được cung cấp.

Quy tắc:
1. Văn phong: Trả lời tự nhiên, lưu loát, lịch sự và mang tính chuyên môn pháp lý cao.
2. Tổng hợp: Hãy diễn đạt lại thông tin một cách thông minh, kết nối các ý lại với nhau thay vì chỉ copy-paste y hệt câu chữ gốc.
3. Trích dẫn: Sau khi đưa ra thông tin, hãy chú thích nguồn ở cuối ý bằng định dạng (Tham chiếu: [Mệnh đề X]).
4. Tính chính xác: Chỉ trả lời dựa trên ngữ cảnh. Nếu ngữ cảnh không có, hãy nói "Hợp đồng hiện tại không quy định chi tiết về vấn đề này." 

DỮ LIỆU CẤU TRÚC TỪ ASSIGNMENT 1 & 2:
{full_context}"""

    # 4. Gọi Ollama LLM
    steps.append(f"generate_with_{OLLAMA_MODEL}")
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.query}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512}
            },
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"]
    except Exception as e:
        logger.error(f"Lỗi gọi Ollama: {str(e)}")
        answer = f"Lỗi kết nối Ollama: {str(e)}. Đảm bảo Ollama đang chạy tại {OLLAMA_BASE}."

    return QueryResponse(
        answer=answer,
        sources=sources,
        pipeline_steps=steps,
        processing_time=time.time() - start_time
    )

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "ollama_base": OLLAMA_BASE,
        "model": OLLAMA_MODEL,
        "vectors_loaded": len(vs.metadata)
    }

@app.delete("/reset")
def reset_system():
    vs.reset()
    return {"message": "Đã xóa toàn bộ cơ sở dữ liệu vector."}

# Khởi động server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
