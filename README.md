#Clone repo về máy: https://github.com/Tritapcode-K23-BKU/Legal-Contract-Analysis.git
#  Legal Contract Analysis — NLP Assignments

Bộ 3 bài tập lớn phân tích hợp đồng pháp lý tiếng Việt sử dụng các kỹ thuật NLP.

---

##  Cấu trúc thư mục

```
Legal-Contract-Analysis-main/
├── assigment1/                         
│   ├── input/
│   │   └── raw_contracts.txt           # Văn bản hợp đồng đầu vào
│   ├── output/                         # Kết quả sinh ra
│   └── src/
│       └── ass1.py                     # Code chính Assignment 1
├── assignment2/
│   ├── input/
│   │   ├── clauses.txt                 # Đầu vào (output từ Assignment 1)
│   │   └── train_dataset_large.json   # Dataset huấn luyện Intent Classification
│   ├── output/                         # Kết quả NER, SRL, Intent
│   └── src/
│       ├── main.py                     # Entry point pipeline Assignment 2
│       ├── task21_ner.py               # Named Entity Recognition
│       ├── task22_srl.py               # Semantic Role Labeling
│       └── task23_intent.py            # Intent Classification (có thể chạy riêng)
└── assignment3/
    ├── backend/
    │   └── app.py                      # FastAPI backend (RAG system)
    └── frontend/
        └── index.html                  # Giao diện web LexRAG
```

---

##  Yêu cầu hệ thống

- **Python** 3.8+
- **Java** 8+ (bắt buộc cho VnCoreNLP ở Assignment 1)
- **Ollama** (bắt buộc cho Assignment 3)
- RAM khuyến nghị: 8GB+

###  Cài đặt nhanh toàn bộ thư viện (Khuyên dùng)
Thay vì cài lẻ tẻ từng bài, bạn có thể cài đặt toàn bộ thư viện của dự án chỉ với một lệnh duy nhất tại thư mục gốc:

```bash
pip install -r requirements.txt

---

##  Assignment 1 — Phân tích cú pháp hợp đồng

**Chức năng:** Tách mệnh đề, gán nhãn IOB (Noun Chunking), phân tích cú pháp phụ thuộc (Dependency Parsing) cho văn bản hợp đồng tiếng Việt.

### Cài đặt thư viện
Vì mô hình PhoBERT và VnCoreNLP vượt quá giới hạn của GitHub, nhóm đã lưu trữ trên Google Drive. 
Thầy/Cô vui lòng thực hiện các bước sau để chạy code:
1. Tải file models.zip tại link sau: https://drive.google.com/file/d/1vXrGP39i82RJrKfeH7tx9BkYhEQCdCQk/view?usp=sharing
2. Giải nén và đặt thư mục `models` ngang hàng với `assignment2/src`.
3. Đặt thư mục `vncorenlp` vào trong `assigment1/src/`.

### Chạy chương trình

```bash
cd assigment1
python src/ass1.py
```

### Kết quả đầu ra (thư mục `assigment1/output/`)

| File | Nội dung |
|------|----------|
| `clauses.txt` | Danh sách mệnh đề đã tách |
| `chunks.txt` | Nhãn IOB (B-NP / I-NP / O) cho từng token |
| `dependency.json` | Cây phụ thuộc cú pháp cho từng mệnh đề |

---

## Assignment 2 — NER + SRL + Intent Classification

**Chức năng:** Pipeline NLP đầy đủ gồm nhận dạng thực thể có tên (NER), gán nhãn vai trò ngữ nghĩa (SRL) và phân loại ý định (Intent Classification) cho các điều khoản hợp đồng.

>  **Đầu vào:** `assignment2/input/clauses.txt` — có thể lấy từ output của Assignment 1.

### Cài đặt thư viện

```bash
pip install scikit-learn numpy

# Tuỳ chọn — tăng độ chính xác NER:
pip install spacy

# Tuỳ chọn — dùng PhoBERT cho Intent Classification:
pip install torch transformers datasets evaluate accelerate
```

### Chạy toàn bộ pipeline (`main.py`)

```bash
cd assignment2

# Chạy tất cả (mặc định)
python src/main.py

# Chạy từng task riêng lẻ
python src/task21_ner.py
python src/task22_srl.py
python src/task23_intent.py
```

### Chạy riêng Intent Classification với dataset lớn (`task23_intent.py`)

File `task23_intent.py` có thể chạy độc lập và hỗ trợ nạp thêm dataset huấn luyện:

```bash
cd assignment2

# Chạy với train_dataset_large.json
python src/task23_intent.py --dataset input/train_dataset_large.json

# Chạy + fine-tune PhoBERT (cần GPU, ~1–2 phút trên RTX 3060)
python src/task23_intent.py --train-phobert --dataset input/train_dataset_large.json

# Tuỳ chỉnh thêm
python task23_intent.py \
    --dataset ../input/train_dataset_large.json \
    --train-phobert \
    --epochs 10 \
    --lr 2e-5 \
    --phobert models/phobert-intent
```

| Flag | Mô tả | Mặc định |
|------|-------|----------|
| `--dataset` | Path đến JSON dataset bổ sung | `None` (dùng data tích hợp) |
| `--train-phobert` | Fine-tune PhoBERT | Tắt |
| `--epochs` | Số epochs fine-tune | `10` |
| `--lr` | Learning rate | `2e-5` |
| `--phobert` | Path model PhoBERT đã fine-tune | `models/phobert-intent` |

### Kết quả đầu ra (thư mục `assignment2/output/`)

| File | Nội dung |
|------|----------|
| `ner_results.json` | Thực thể nhận dạng (PARTY, MONEY, DATE, RATE, PENALTY, LAW, LOCATION) |
| `srl_results.json` | Khung ngữ nghĩa và vai trò các tham số |
| `intent_classification.txt` | Ý định phân loại dạng text |
| `intent_classification.json` | Ý định phân loại dạng JSON |
| `model_comparison.json` | So sánh các mô hình |
| `combined_results.json` | Tổng hợp kết quả cả 3 task |

---

##  Assignment 3 — LexRAG: Hệ thống hỏi đáp hợp đồng (RAG)

**Chức năng:** Hệ thống RAG (Retrieval-Augmented Generation) kết hợp kết quả từ Assignment 1 & 2 để trả lời câu hỏi về nội dung hợp đồng thông qua giao diện web.

### Yêu cầu bổ sung

**Cài đặt Ollama** (LLM backend):
```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Tải tại https://ollama.com/download
```

**Tải model LLM (mặc định: qwen2.5:7b):**
```bash
ollama pull qwen2.5:7b
```

> Muốn dùng model khác, đặt biến môi trường `OLLAMA_MODEL` (ví dụ: `llama3.2`, `mistral`).

### Cài đặt thư viện Python

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu numpy requests pydantic
```

> Trên máy có GPU: thay `faiss-cpu` bằng `faiss-gpu`.

### Chạy backend

```bash
# Bước 1: Đảm bảo Ollama đang chạy
ollama serve

# Bước 2: Chạy FastAPI backend
cd assignment3/backend
python app.py
```

Backend khởi động tại: **http://localhost:8000**

Kiểm tra trạng thái:
```bash
curl http://localhost:8000/health
```

**Tuỳ chỉnh cấu hình qua biến môi trường:**
```bash
OLLAMA_BASE=http://localhost:11434 \
OLLAMA_MODEL=qwen2.5:7b \
python app.py
```

### Chạy frontend

Mở trực tiếp file HTML trong trình duyệt:
```bash
# macOS
open assignment3/frontend/index.html

# Linux
xdg-open assignment3/frontend/index.html

# Windows
start assignment3/frontend/index.html
```

Hoặc dùng Live Server (VS Code) / HTTP server:
```bash
cd assignment3/frontend
python -m http.server 3000
# Truy cập: http://localhost:3000
```

---

##  Quy trình chạy toàn bộ dự án

Để có kết quả tốt nhất, chạy theo thứ tự:

```
Assignment 1  →  Assignment 2  →  Assignment 3
(Tách mệnh đề)   (NER/SRL/Intent)  (RAG Chatbot)
```

Output `clauses.txt` của Assignment 1 là đầu vào của Assignment 2.
Assignment 3 sử dụng kết quả tổng hợp từ cả hai assignment trước.

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `Java not found` | Java chưa cài | Cài Java 8+ và thêm vào PATH |
| `No module named 'py_vncorenlp'` | Thiếu thư viện | `pip install py-vncorenlp` |
| `Connection refused` (Assignment 3) | Ollama chưa chạy | Chạy `ollama serve` trước |
| `model not found` (Ollama) | Chưa tải model | `ollama pull qwen2.5:7b` |
| `faiss` lỗi trên Windows | Build tools thiếu | `pip install faiss-cpu --no-build-isolation` |
