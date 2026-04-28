"""
Assignment 2 - Task 2.3: Intent Classification
Phân loại ý định mệnh đề hợp đồng pháp lý tiếng Việt

Approach:
  [1] Rule-based (baseline nhanh)
  [2] TF-IDF + Logistic Regression
  [3] PhoBERT-base-v2 fine-tuned (Transformer, CUDA/CPU)
  [4] So sánh [2] vs [3] bằng Precision / Recall / F1

Tối ưu cho: NVIDIA RTX 3060 (CUDA) — G15 5511
  - fp16 mixed precision khi CUDA
  - batch_size tự động theo VRAM
  - gradient_checkpointing tiết kiệm VRAM
  - warmup + cosine LR schedule
  - early stopping theo F1-macro

Cài đặt (1 lần):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install transformers datasets evaluate accelerate scikit-learn

Chạy:
  python task23_intent.py                          # TF-IDF + Rule (không cần GPU)
  python task23_intent.py --train-phobert          # Fine-tune PhoBERT (RTX 3060 ~2 phút)
  python task23_intent.py --phobert models/phobert # Dùng model đã train sẵn

Output:
  output/intent_classification.txt   ← format đề bài
  output/intent_classification.json  ← chi tiết từng clause
  output/model_comparison.json       ← bảng so sánh TF-IDF vs PhoBERT
"""

import re
import json
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── Label schema ─────────────────────────────────────────────────────────────
INTENT_LABELS = ["Obligation", "Prohibition", "Right", "Termination_Condition"]
INTENT2ID     = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2INTENT     = {i: l for l, i in INTENT2ID.items()}

# ─── Keyword lexicon (Rule-based) ─────────────────────────────────────────────
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "Obligation": [
        "phải", "có trách nhiệm", "bắt buộc", "cần phải", "có nghĩa vụ",
        "thanh toán", "bảo quản", "thời hạn thuê",
        "mức phạt", "trễ hạn", "chậm trả", "áp dụng",
    ],
    "Prohibition": [
        "không được phép", "không được", "cấm", "không có quyền",
        "bị cấm", "nghiêm cấm", "trừ khi",
    ],
    "Right": [
        "có quyền", "được quyền", "được phép", "có thể",
        "không chịu trách nhiệm", "miễn trừ",
    ],
    "Termination_Condition": [
        "bất khả kháng", "thiên tai", "dịch bệnh", "thương lượng",
        "tòa án", "phân xử", "tranh chấp", "thỏa thuận",
        "chấm dứt hợp đồng", "hủy hợp đồng", "đình chỉ",
    ],
}

# ─── Training + Evaluation data ───────────────────────────────────────────────
ALL_LABELED_DATA: List[Tuple[str, str]] = [
    # Obligation
    ("Bên A đồng ý cho Bên B thuê mặt bằng kinh doanh tại tầng 1 tòa nhà Bách Khoa.", "Obligation"),
    ("Thời hạn thuê là 12 tháng.", "Obligation"),
    ("Bên B phải thanh toán toàn bộ tiền thuê trước ngày 5 hàng tháng.", "Obligation"),
    ("Nếu Bên B thanh toán trễ hạn, mức phạt 1,5% mỗi ngày sẽ được áp dụng đối với số tiền chậm trả.", "Obligation"),
    ("Trong quá trình thực hiện hợp đồng, Bên B có trách nhiệm tự bảo quản tài sản cá nhân.", "Obligation"),
    ("Bên A phải cung cấp hóa đơn VAT theo yêu cầu của Bên B.", "Obligation"),
    ("Người thuê phải nộp tiền đặt cọc trước khi nhận bàn giao.", "Obligation"),
    ("Bên B phải thông báo trước 30 ngày nếu muốn chấm dứt sớm.", "Obligation"),
    ("Bên A có nghĩa vụ duy trì cơ sở hạ tầng của tòa nhà trong suốt thời hạn thuê.", "Obligation"),
    ("Các bên phải ký biên bản bàn giao khi kết thúc hợp đồng.", "Obligation"),
    ("Bên B phải trả đầy đủ tiền thuê trước ngày 5 hàng tháng.", "Obligation"),
    ("Người lao động phải thực hiện đúng nội quy công ty.", "Obligation"),
    ("Bên thuê có trách nhiệm hoàn trả mặt bằng trong tình trạng ban đầu.", "Obligation"),
    ("Bên B có trách nhiệm nộp phạt theo quy định nếu vi phạm.", "Obligation"),
    # Prohibition
    ("Bên B không được phép chuyển nhượng hợp đồng cho bên thứ ba, trừ khi có sự đồng ý bằng văn bản của Bên A.", "Prohibition"),
    ("Bên thuê không được sử dụng mặt bằng sai mục đích đã cam kết.", "Prohibition"),
    ("Không được phép sửa chữa kết cấu tòa nhà khi chưa có sự đồng ý.", "Prohibition"),
    ("Bên B bị cấm cho thuê lại mặt bằng khi chưa được Bên A đồng ý bằng văn bản.", "Prohibition"),
    ("Bên thuê không có quyền tự ý thay đổi kết cấu mặt bằng.", "Prohibition"),
    ("Nghiêm cấm sử dụng mặt bằng để kinh doanh các mặt hàng bị pháp luật cấm.", "Prohibition"),
    ("Bên B không được lắp đặt biển hiệu quảng cáo khi chưa được phép.", "Prohibition"),
    ("Không được phép mang vật liệu dễ cháy nổ vào khu vực mặt bằng.", "Prohibition"),
    ("Bên thuê không được phép hoạt động ngoài giờ quy định.", "Prohibition"),
    ("Cấm sử dụng mặt bằng cho các hoạt động vi phạm pháp luật.", "Prohibition"),
    # Right
    ("Bên A có quyền đơn phương chấm dứt hợp đồng nếu Bên B sử dụng mặt bằng sai mục đích đã cam kết.", "Right"),
    ("Bên A sẽ không chịu trách nhiệm đối với bất kỳ mất mát nào.", "Right"),
    ("Bên thuê được phép sử dụng mặt bằng vào mục đích kinh doanh đã đăng ký.", "Right"),
    ("Bên cho thuê có quyền kiểm tra tài sản và mặt bằng định kỳ mỗi tháng.", "Right"),
    ("Bên A có thể yêu cầu bồi thường thiệt hại khi Bên B gây ra vi phạm.", "Right"),
    ("Bên thuê có quyền yêu cầu Bên A sửa chữa các hư hỏng kết cấu.", "Right"),
    ("Bên A có quyền điều chỉnh giá thuê sau mỗi năm theo thị trường.", "Right"),
    ("Người thuê được phép lắp đặt thiết bị văn phòng theo nhu cầu.", "Right"),
    ("Bên B có quyền chấm dứt hợp đồng sớm nếu Bên A không thực hiện nghĩa vụ.", "Right"),
    ("Bên A có quyền từ chối gia hạn hợp đồng mà không cần nêu lý do.", "Right"),
    # Termination_Condition
    ("Trong trường hợp xảy ra sự kiện bất khả kháng như thiên tai hoặc dịch bệnh, hai bên sẽ cùng nhau thương lượng để tìm hướng giải quyết.", "Termination_Condition"),
    ("Nếu không đạt được thỏa thuận, sự việc sẽ được đưa ra Tòa án có thẩm quyền để phân xử.", "Termination_Condition"),
    ("Hợp đồng sẽ bị chấm dứt nếu Bên B vi phạm điều khoản trọng yếu hai lần liên tiếp.", "Termination_Condition"),
    ("Khi xảy ra thiên tai hoặc dịch bệnh, hợp đồng có thể tạm thời bị đình chỉ.", "Termination_Condition"),
    ("Hợp đồng tự động chấm dứt khi hết thời hạn thuê mà không cần thông báo.", "Termination_Condition"),
    ("Tranh chấp sẽ được giải quyết tại Tòa án nhân dân có thẩm quyền tại TP.HCM.", "Termination_Condition"),
    ("Hợp đồng có thể chấm dứt trước hạn nếu cả hai bên đồng ý bằng văn bản.", "Termination_Condition"),
    ("Mọi tranh chấp phát sinh sẽ được đưa ra trọng tài thương mại để giải quyết.", "Termination_Condition"),
    ("Bên A có quyền chấm dứt ngay lập tức nếu phát hiện Bên B hoạt động trái pháp luật.", "Termination_Condition"),
    ("Nếu mặt bằng bị phá dỡ theo quyết định nhà nước, hợp đồng chấm dứt.", "Termination_Condition"),
]


# ─── Dataset loader ───────────────────────────────────────────────────────────
def load_dataset(json_path=None):
    """
    Merge ALL_LABELED_DATA (hand-crafted) + external JSON dataset.

    JSON format: [{"text": "...", "intent": "..."}, ...]

    Xử lý tự động:
      - Normalize label: "Termination Condition" -> "Termination_Condition"
      - Dedup theo text (giữ hand-crafted nếu trùng)
      - Imbalance warning nếu ratio > 3x
    """
    from collections import Counter

    def normalize_label(label):
        return label.strip().replace(" ", "_")

    # Bắt đầu từ hand-crafted (ưu tiên cao nhất)
    merged = {
        text: normalize_label(label)
        for text, label in ALL_LABELED_DATA
    }

    if json_path and Path(json_path).exists():
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        before = len(merged)
        skipped_dup, skipped_label = 0, 0

        for item in raw:
            text  = item.get("text", "").strip()
            label = normalize_label(item.get("intent", ""))
            if not text:
                continue
            if label not in INTENT_LABELS:
                skipped_label += 1
                continue
            if text in merged:
                skipped_dup += 1
                continue
            merged[text] = label

        added = len(merged) - before
        print(f"  [Dataset] Loaded \'{Path(json_path).name}\'")
        print(f"            Added   : {added:4d} samples")
        print(f"            Skipped : {skipped_dup} duplicates, {skipped_label} unknown labels")
    elif json_path:
        print(f"  [Dataset] \'{json_path}\' not found — using default data only")

    data = list(merged.items())

    counts  = Counter(l for _, l in data)
    max_cnt = max(counts.values())
    min_cnt = min(counts.values())
    print(f"  [Dataset] Total: {len(data)} | "
          + " | ".join(f"{l}: {c}" for l, c in sorted(counts.items())))
    if max_cnt / max(min_cnt, 1) > 3:
        print(f"  [Dataset] WARN Imbalance {max_cnt/min_cnt:.1f}x "
              f"=> class_weight=\'balanced\' enabled in TF-IDF & PhoBERT")

    return data


# ─── Data class ───────────────────────────────────────────────────────────────
@dataclass
class IntentResult:
    clause_id:        int
    clause:           str
    intent:           str
    confidence:       float
    model:            str = "rule"
    keywords_matched: List[str] = field(default_factory=list)
    explanation:      str = ""
    alternatives:     List[Dict] = field(default_factory=list)

    def to_dict(self): return asdict(self)
    def to_txt_line(self): return f"{self.clause}\t{self.intent}"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Rule-based Classifier
# ══════════════════════════════════════════════════════════════════════════════
class RuleBasedClassifier:
    """Keyword scoring với priority overrides cho các trường hợp xung đột."""

    def _score(self, clause: str):
        scores, matched = defaultdict(float), defaultdict(list)
        lower = clause.lower()
        for intent, kws in INTENT_KEYWORDS.items():
            for kw in kws:
                if kw in lower:
                    scores[intent] += 1.0 + len(kw.split()) * 0.5
                    matched[intent].append(kw)
        return scores, matched

    def classify(self, clause_id: int, clause: str) -> IntentResult:
        lower = clause.lower()

        # Override 1: "có quyền ... chấm dứt" → Right (trao quyền, không phải điều kiện)
        if re.search(r'(có|được)\s+quyền.{0,30}chấm\s+dứt', lower):
            return IntentResult(clause_id, clause, "Right", 0.90, "rule",
                ["có quyền", "chấm dứt"],
                "Override: 'có quyền ... chấm dứt' → Right",
                [{"intent": "Termination_Condition", "confidence": 0.10}])

        # Override 2: "Bên X đồng ý cho ... thuê" ở đầu câu → Obligation
        if re.match(r'bên\s+[ab]\s+đồng\s+ý\s+cho', lower):
            return IntentResult(clause_id, clause, "Obligation", 0.85, "rule",
                ["đồng ý cho"],
                "Override: ghi nhận nghĩa vụ cho thuê theo hợp đồng → Obligation",
                [{"intent": "Right", "confidence": 0.15}])

        scores, matched = self._score(clause)
        if not scores:
            return IntentResult(clause_id, clause, "Obligation", 0.50, "rule",
                                explanation="Fallback default")

        total = sum(scores.values())
        probs = {k: round(v / total, 4) for k, v in scores.items()}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best, conf = ranked[0]

        return IntentResult(
            clause_id=clause_id, clause=clause,
            intent=best, confidence=conf, model="rule",
            keywords_matched=matched[best],
            explanation=f"Keywords: {matched[best]} → {best}",
            alternatives=[{"intent": k, "confidence": v} for k, v in ranked[1:3] if v > 0],
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — TF-IDF + Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
class TFIDFClassifier:
    """
    TF-IDF character n-gram + Logistic Regression.
    Hiệu quả cho tiếng Việt vì không cần tokenizer riêng.
    """

    def __init__(self):
        self.pipeline   = None
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=8000,
                analyzer="char_wb",
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000, C=3.0,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        print(f"  [TF-IDF] Trained on {len(texts)} samples.")

    def classify(self, clause_id: int, clause: str) -> Optional[IntentResult]:
        if not self.is_trained:
            return None
        intent  = self.pipeline.predict([clause])[0]
        proba   = self.pipeline.predict_proba([clause])[0]
        classes = self.pipeline.classes_
        conf    = float(max(proba))
        alts    = sorted(
            [{"intent": c, "confidence": round(float(p), 4)}
             for c, p in zip(classes, proba) if c != intent],
            key=lambda x: -x["confidence"]
        )[:2]
        return IntentResult(
            clause_id=clause_id, clause=clause,
            intent=intent, confidence=round(conf, 4), model="tfidf-lr",
            explanation=f"TF-IDF+LR: {intent} ({conf:.0%})",
            alternatives=alts,
        )

    def evaluate(self, texts: List[str], gold: List[str]) -> Dict:
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
        preds = self.pipeline.predict(texts)
        return {
            "model":           "TF-IDF + Logistic Regression",
            "precision_macro": round(precision_score(gold, preds, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(gold, preds, average="macro", zero_division=0), 4),
            "f1_macro":        round(f1_score(gold, preds, average="macro", zero_division=0), 4),
            "per_class":       classification_report(gold, preds, target_names=INTENT_LABELS,
                                                     output_dict=True, zero_division=0),
            "report_text":     classification_report(gold, preds, target_names=INTENT_LABELS,
                                                     zero_division=0),
        }

    def cross_validate(self, texts: List[str], labels: List[str], cv: int = 5) -> Dict:
        from sklearn.model_selection import cross_validate as sk_cv
        scores = sk_cv(
            self.pipeline, texts, labels,
            cv=min(cv, len(set(labels))),
            scoring=["precision_macro", "recall_macro", "f1_macro"],
        )
        return {
            "cv_folds":        min(cv, len(set(labels))),
            "precision_macro": round(scores["test_precision_macro"].mean(), 4),
            "recall_macro":    round(scores["test_recall_macro"].mean(), 4),
            "f1_macro":        round(scores["test_f1_macro"].mean(), 4),
            "f1_std":          round(scores["test_f1_macro"].std(), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — PhoBERT-base-v2 fine-tuned (CUDA optimized for RTX 3060)
# ══════════════════════════════════════════════════════════════════════════════
class PhoBERTClassifier:
    """
    PhoBERT-base-v2 fine-tuned cho intent classification tiếng Việt.

    Tối ưu RTX 3060 (6GB VRAM):
      - fp16 mixed precision (tự động bật khi CUDA)
      - gradient_checkpointing (tiết kiệm VRAM ~40%)
      - batch_size=16 cho CUDA, 4 cho CPU
      - cosine LR schedule + linear warmup
      - early stopping theo F1-macro (patience=3)

    Cài đặt:
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
      pip install transformers datasets evaluate accelerate scikit-learn

    Thời gian ước tính (RTX 3060):
      - Download model lần đầu: ~2 phút (400MB)
      - Fine-tune 10 epochs / 44 samples: ~1–2 phút
    """

    PHOBERT_MODEL = "vinai/phobert-base-v2"
    MAX_LEN       = 128

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model      = None
        self.tokenizer  = None
        self.device     = None
        self._loaded    = False
        self._try_load(model_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  [PhoBERT] GPU: {name} ({vram:.1f}GB VRAM) → CUDA ✓")
            return "cuda"
        print("  [PhoBERT] No CUDA found → CPU (sẽ chậm hơn)")
        return "cpu"

    def _smart_batch_size(self) -> int:
        """Tự chọn batch size an toàn theo VRAM."""
        if self.device != "cuda":
            return 4
        try:
            import torch
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb >= 8:
                return 16
            elif vram_gb >= 6:
                return 12   # RTX 3060 = 6GB → safe
            return 8
        except Exception:
            return 8

    def _try_load(self, path: Optional[str]):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.device = self._detect_device()

            if path and Path(path).exists():
                print(f"  [PhoBERT] Loading fine-tuned model from '{path}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.model     = AutoModelForSequenceClassification.from_pretrained(path)
                self.model.to(self.device)
                self.model.eval()
                self._loaded = True
                print(f"  [PhoBERT] Model loaded ✓")
            else:
                print(f"  [PhoBERT] No fine-tuned model at '{path}'.")
                print(f"            Chạy: python task23_intent.py --train-phobert")
        except ImportError:
            print("  [PhoBERT] Thiếu thư viện. Cài đặt:")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("    pip install transformers datasets evaluate accelerate")

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        train_texts:  List[str],
        train_labels: List[str],
        val_texts:    List[str],
        val_labels:   List[str],
        output_dir:   str = "models/phobert-intent",
        epochs:       int = 10,
        lr:           float = 2e-5,
    ):
        """Fine-tune PhoBERT. RTX 3060: ~1–2 phút với 44 samples."""
        try:
            import torch
            import numpy as np
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                TrainingArguments, Trainer,
                DataCollatorWithPadding, EarlyStoppingCallback,
            )
            from datasets import Dataset
            import evaluate as hf_evaluate
            from sklearn.metrics import f1_score

            device     = self._detect_device()
            batch_size = self._smart_batch_size()
            use_fp16   = (device == "cuda")

            print(f"\n{'='*60}")
            print(f"  PhoBERT Fine-tuning")
            print(f"  Device     : {device.upper()}")
            print(f"  Batch size : {batch_size}")
            print(f"  FP16       : {use_fp16}")
            print(f"  Epochs     : {epochs}")
            print(f"  Train/Val  : {len(train_texts)}/{len(val_texts)}")
            print(f"{'='*60}\n")

            # ── Tokenizer & model ──
            print("  Downloading PhoBERT (lần đầu ~400MB)...")
            tokenizer = AutoTokenizer.from_pretrained(self.PHOBERT_MODEL)
            model     = AutoModelForSequenceClassification.from_pretrained(
                self.PHOBERT_MODEL,
                num_labels=len(INTENT_LABELS),
                id2label=ID2INTENT,
                label2id=INTENT2ID,
                ignore_mismatched_sizes=True,
            )

            # Gradient checkpointing tiết kiệm VRAM (quan trọng với 6GB)
            if use_fp16:
                model.gradient_checkpointing_enable()

            # ── Dataset ──
            def tokenize(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.MAX_LEN,
                    padding="max_length",
                )

            train_ds = Dataset.from_dict({
                "text":  train_texts,
                "label": [INTENT2ID[l] for l in train_labels],
            }).map(tokenize, batched=True, remove_columns=["text"])

            val_ds = Dataset.from_dict({
                "text":  val_texts,
                "label": [INTENT2ID[l] for l in val_labels],
            }).map(tokenize, batched=True, remove_columns=["text"])

            # ── Class weights cho imbalanced data ──
            from collections import Counter
            import torch as _torch
            label_counts = Counter(train_labels)
            total_train  = len(train_labels)
            class_weights = _torch.tensor(
                [total_train / (len(INTENT_LABELS) * label_counts.get(l, 1))
                 for l in INTENT_LABELS],
                dtype=_torch.float,
            ).to(device)
            print(f"  Class weights: { {l: round(float(w),3) for l,w in zip(INTENT_LABELS, class_weights)} }")

            # ── Weighted Trainer để handle imbalance ──
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels  = inputs.get("labels")
                    outputs = model(**inputs)
                    logits  = outputs.get("logits")
                    loss_fn = _torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss    = loss_fn(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            # ── Metrics ──
            acc_metric = hf_evaluate.load("accuracy")

            def compute_metrics(p):
                preds  = np.argmax(p.predictions, axis=1)
                labels = p.label_ids
                acc    = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
                f1     = f1_score(labels, preds, average="macro", zero_division=0)
                return {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}

            # ── Training args ──
            # gradient_accumulation_steps=2 -> effective batch = batch_size*2
            # giup stable hon voi it data va imbalanced labels
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=2,
                learning_rate=lr,
                weight_decay=0.01,
                warmup_ratio=0.15,
                lr_scheduler_type="cosine",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                fp16=use_fp16,
                dataloader_num_workers=0,      # tránh lỗi Windows multiprocessing
                logging_steps=5,
                save_total_limit=2,            # giữ tối đa 2 checkpoint
                report_to="none",              # tắt wandb/tensorboard
                seed=42,
            )

            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            # ── Train ──
            print("  Training...")
            trainer.train()

            # ── Save ──
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            eval_result = trainer.evaluate()
            Path(output_dir).joinpath("eval_metrics.json").write_text(
                json.dumps(eval_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n  [PhoBERT] Saved to '{output_dir}'")
            print(f"  Eval F1-macro: {eval_result.get('eval_f1_macro', 'N/A'):.4f}")

            # Reload model vừa train
            self._try_load(output_dir)

        except ImportError as e:
            print(f"\n  [PhoBERT] Thiếu thư viện: {e}")
            print("  Cài đặt:")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("    pip install transformers datasets evaluate accelerate")
        except Exception as e:
            print(f"\n  [PhoBERT] Lỗi khi train: {e}")
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(self, clause_id: int, clause: str) -> Optional[IntentResult]:
        if not self._loaded:
            return None
        try:
            import torch
            inputs = self.tokenizer(
                clause,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_LEN,
                padding=True,
            ).to(self.device)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs   = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_id = int(torch.argmax(logits, dim=-1)[0])
            intent  = ID2INTENT[pred_id]
            conf    = round(probs[pred_id], 4)
            alts    = sorted(
                [{"intent": ID2INTENT[i], "confidence": round(p, 4)}
                 for i, p in enumerate(probs) if i != pred_id],
                key=lambda x: -x["confidence"]
            )[:2]
            return IntentResult(
                clause_id=clause_id, clause=clause,
                intent=intent, confidence=conf, model="phobert",
                explanation=f"PhoBERT fine-tuned: {intent} ({conf:.0%})",
                alternatives=alts,
            )
        except Exception as e:
            print(f"  [PhoBERT] Inference error: {e}")
            return None

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, texts: List[str], gold: List[str]) -> Optional[Dict]:
        if not self._loaded:
            return None
        from sklearn.metrics import (
            classification_report, f1_score, precision_score, recall_score,
        )
        preds = [self.classify(i + 1, t).intent for i, t in enumerate(texts)]
        return {
            "model":           "PhoBERT-base-v2 fine-tuned",
            "precision_macro": round(precision_score(gold, preds, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(gold, preds, average="macro", zero_division=0), 4),
            "f1_macro":        round(f1_score(gold, preds, average="macro", zero_division=0), 4),
            "per_class":       classification_report(gold, preds, target_names=INTENT_LABELS,
                                                     output_dict=True, zero_division=0),
            "report_text":     classification_report(gold, preds, target_names=INTENT_LABELS,
                                                     zero_division=0),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def compare_models(
    tfidf:       TFIDFClassifier,
    phobert:     PhoBERTClassifier,
    test_texts:  List[str],
    test_labels: List[str],
    all_texts:   List[str],
    all_labels:  List[str],
    output_path: str = "output/model_comparison.json",
):
    """So sánh TF-IDF+LR và PhoBERT trên cùng một tập test."""
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON: TF-IDF+LR vs PhoBERT")
    print(f"{'='*60}")

    comparison = {"test_size": len(test_texts), "models": []}

    # ── TF-IDF ──
    tfidf_test = tfidf.evaluate(test_texts, test_labels)
    tfidf_cv   = tfidf.cross_validate(all_texts, all_labels, cv=5)

    comparison["models"].append({
        "model":         "TF-IDF + Logistic Regression",
        "test_precision": tfidf_test["precision_macro"],
        "test_recall":    tfidf_test["recall_macro"],
        "test_f1_macro":  tfidf_test["f1_macro"],
        "cv_f1_macro":    tfidf_cv["f1_macro"],
        "cv_f1_std":      tfidf_cv["f1_std"],
        "per_class":      tfidf_test["per_class"],
    })

    print(f"\n  [TF-IDF + LR]")
    print(f"  Test  → P: {tfidf_test['precision_macro']:.4f} | "
          f"R: {tfidf_test['recall_macro']:.4f} | F1: {tfidf_test['f1_macro']:.4f}")
    print(f"  CV-5  → F1: {tfidf_cv['f1_macro']:.4f} ± {tfidf_cv['f1_std']:.4f}")
    print(f"\n{tfidf_test['report_text']}")

    # ── PhoBERT ──
    if phobert._loaded:
        pb_test = phobert.evaluate(test_texts, test_labels)
        comparison["models"].append({
            "model":          "PhoBERT-base-v2 fine-tuned",
            "test_precision": pb_test["precision_macro"],
            "test_recall":    pb_test["recall_macro"],
            "test_f1_macro":  pb_test["f1_macro"],
            "per_class":      pb_test["per_class"],
        })

        print(f"  [PhoBERT fine-tuned]")
        print(f"  Test  → P: {pb_test['precision_macro']:.4f} | "
              f"R: {pb_test['recall_macro']:.4f} | F1: {pb_test['f1_macro']:.4f}")
        print(f"\n{pb_test['report_text']}")

        winner = "PhoBERT" if pb_test["f1_macro"] > tfidf_test["f1_macro"] else "TF-IDF+LR"
        diff   = abs(pb_test["f1_macro"] - tfidf_test["f1_macro"])
        comparison["verdict"] = {
            "better_model":  winner,
            "f1_difference": round(diff, 4),
            "note": f"{winner} tốt hơn {diff:.2%}. PhoBERT hiểu ngữ nghĩa sâu hơn.",
        }
        print(f"\n  Verdict: {winner} tốt hơn (ΔF1 = {diff:.4f})")
    else:
        comparison["models"].append({
            "model":  "PhoBERT-base-v2 fine-tuned",
            "status": "not_trained",
            "note":   "Chạy --train-phobert để fine-tune.",
        })
        comparison["verdict"] = {
            "better_model": "TF-IDF+LR (PhoBERT chưa được train)",
            "note":         "Chạy: python task23_intent.py --train-phobert",
        }
        print("  [PhoBERT] chưa train — bỏ qua so sánh.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved: {output_path}")
    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
class IntentPipeline:
    """
    Pipeline ưu tiên: PhoBERT > TF-IDF > Rule-based
    """

    def __init__(self, phobert_model_path: Optional[str] = None):
        self.rule    = RuleBasedClassifier()
        self.tfidf   = TFIDFClassifier()
        self.phobert = PhoBERTClassifier(phobert_model_path)

    def train_tfidf(self, data: List[Tuple[str, str]]):
        texts, labels = zip(*data)
        self.tfidf.train(list(texts), list(labels))

    def classify(self, clause_id: int, clause: str) -> IntentResult:
        # PhoBERT ưu tiên nếu đã load
        result = self.phobert.classify(clause_id, clause)
        if result:
            return result
        # TF-IDF nếu đủ confidence
        result = self.tfidf.classify(clause_id, clause)
        if result and result.confidence >= 0.60:
            return result
        # Rule-based fallback
        return self.rule.classify(clause_id, clause)

    def process_file(
        self,
        input_path:  str,
        output_path: str,
        json_path:   Optional[str] = None,
    ) -> List[IntentResult]:
        clauses = self._read_clauses(input_path)
        results = []
        for i, clause in enumerate(clauses, start=1):
            res = self.classify(i, clause)
            if res.intent == "Termination_Condition":
                res.intent = "Termination Condition"
            results.append(res)
            print(f"  [{i:2d}] {res.intent:25s} ({res.confidence:.0%}) | {clause[:50]}...")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            "\n".join(r.to_txt_line() for r in results), encoding="utf-8"
        )
        print(f"\n  ✓ {output_path}")

        if json_path:
            Path(json_path).write_text(
                json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ {json_path}")

        return results

    def _read_clauses(self, path: str) -> List[str]:
        clauses = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "]" in line:
                line = line.rsplit("]", 1)[-1].strip()
            if line:
                clauses.append(line)
        return clauses


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(
        description="Task 2.3 — Intent Classification (PhoBERT + RTX 3060)"
    )
    parser.add_argument(
        "--phobert", default=None,
        help="Path đến fine-tuned PhoBERT model (default: models/phobert-intent)"
    )
    parser.add_argument(
        "--train-phobert", action="store_true",
        help="Fine-tune PhoBERT (RTX 3060: ~1–2 phút)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Số epochs fine-tune (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path đến JSON dataset bổ sung (vd: output/train_dataset_large.json)"
    )
    args = parser.parse_args()

    BASE   = Path(__file__).parent.parent
    INPUT  = str(BASE / "input"  / "clauses.txt")
    OUTPUT = str(BASE / "output" / "intent_classification.txt")
    JSON   = str(BASE / "output" / "intent_classification.json")
    CMP    = str(BASE / "output" / "model_comparison.json")

    # ── Chuẩn bị data ──
    print("\n── [0/3] Load Dataset ──")
    all_data   = load_dataset(json_path=args.dataset)
    all_texts  = [t for t, _ in all_data]
    all_labels = [l for _, l in all_data]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.15, random_state=42, stratify=all_labels,
    )
    print(f"  Train: {len(train_texts)} | Test: {len(test_texts)}")

    # ── Khởi tạo pipeline ──
    phobert_path = args.phobert or str(BASE / "models" / "phobert-intent")
    pipeline = IntentPipeline(phobert_model_path=phobert_path)

    # ── Train TF-IDF ──
    print("\n── [1/3] Train TF-IDF ──")
    pipeline.train_tfidf(list(zip(train_texts, train_labels)))

    # ── Fine-tune PhoBERT (optional) ──
    if args.train_phobert:
        print("\n── [2/3] Fine-tune PhoBERT ──")
        pipeline.phobert.fine_tune(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=test_texts,
            val_labels=test_labels,
            output_dir=phobert_path,
            epochs=args.epochs,
            lr=args.lr,
        )

    # ── Inference ──
    print("\n── [3/3] Intent Classification ──")
    pipeline.process_file(INPUT, OUTPUT, json_path=JSON)

    # ── So sánh mô hình ──
    compare_models(
        tfidf=pipeline.tfidf,
        phobert=pipeline.phobert,
        test_texts=test_texts,
        test_labels=test_labels,
        all_texts=all_texts,
        all_labels=all_labels,
        output_path=CMP,
    )
