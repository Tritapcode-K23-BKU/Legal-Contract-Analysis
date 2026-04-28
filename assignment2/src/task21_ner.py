"""
Assignment 2 - Task 2.1: Named Entity Recognition (NER)
Custom NER for Vietnamese Legal Contracts
Approach: Rule-based baseline + spaCy PhoBERT fine-tuning
Output: output/ner_results.json
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


# ─── Entity Schema ────────────────────────────────────────────────────────────
# Bổ sung thêm nhãn LOCATION theo ý bạn
ENTITY_LABELS = {
    "PARTY":    "Bên ký kết hợp đồng (Bên A, Bên B, bên thứ ba...)",
    "MONEY":    "Số tiền, giá trị tài chính",
    "DATE":     "Ngày, thời hạn cụ thể",
    "RATE":     "Tỷ lệ phần trăm, lãi suất, mức phạt %",
    "PENALTY":  "Điều khoản phạt vi phạm",
    "LAW":      "Điều luật, quy định pháp lý, hợp đồng",
    "LOCATION": "Địa điểm, mặt bằng, tòa nhà, trụ sở",
}


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int
    description: str = ""

@dataclass
class NERResult:
    clause_id: int
    clause: str
    entities: List[Entity]

    def to_dict(self):
        return {
            "clause_id": self.clause_id,
            "clause": self.clause,
            "entities": [asdict(e) for e in self.entities]
        }


# ─── Rule-based NER (Baseline) ────────────────────────────────────────────────
class RuleBasedNER:
    """
    Rule-based NER sử dụng regex patterns cho hợp đồng pháp lý tiếng Việt.
    """

    PATTERNS: List[Tuple[str, str, str]] = [
        # PARTY
        ("PARTY",    r"\bBên\s+[A-Z]\b", "Bên ký kết"),
        ("PARTY",    r"bên\s+thứ\s+ba", "Bên thứ ba"),
        ("PARTY",    r"hai\s+bên", "Hai bên ký kết"),

        # MONEY
        ("MONEY",    r"tiền\s+thuê", "Tiền thuê"),
        ("MONEY",    r"số\s+tiền\s+chậm\s+trả", "Tiền phạt chậm"),
        ("MONEY",    r"\d{1,3}(?:[.,]\d{3})*(?:\s*(?:VNĐ|VND|đồng|triệu|tỷ))", "Số tiền"),

        # DATE 
        ("DATE",     r"\d+\s+tháng", "Thời hạn"),
        ("DATE",     r"trước\s+ngày\s+\d+\s+hàng\s+tháng", "Hạn thanh toán hàng tháng"),
        ("DATE",     r"ngày\s+\d{1,2}(?:\s*[/\-]\s*\d{1,2}(?:\s*[/\-]\s*\d{2,4})?)?", "Ngày cụ thể"),

        # RATE
        ("RATE",     r"\d+(?:[,\.]\d+)?\s*%", "Tỷ lệ phần trăm"),

        # PENALTY
        ("PENALTY",  r"mức\s+phạt", "Mức phạt"),
        ("PENALTY",  r"trễ\s+hạn", "Trễ hạn"),
        ("PENALTY",  r"sự\s+kiện\s+bất\s+khả\s+kháng", "Sự kiện bất khả kháng"),

        # LAW
        ("LAW",      r"hợp\s+đồng", "Văn bản hợp đồng"),
        ("LAW",      r"văn\s+bản", "Hình thức văn bản"),
        ("LAW",      r"Tòa\s+án(?:[^,\.]*)?", "Cơ quan pháp luật"),

        # LOCATION (Đã thêm vào để bắt địa điểm)
        ("LOCATION", r"mặt\s+bằng\s+kinh\s+doanh\s+tại\s+tầng\s+\d+\s+tòa\s+nhà\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "Địa điểm chi tiết"),
        ("LOCATION", r"mặt\s+bằng\s+sai\s+mục\s+đích\s+đã\s+cam\s+kết", "Mặt bằng"),
        ("LOCATION", r"(?:tầng\s+\d+\s+)?(?:tòa\s+nhà|mặt\s+bằng|văn\s+phòng|trụ\s+sở)[^,\.]*", "Địa điểm cơ bản"),
    ]

    def __init__(self):
        self._compiled = [
            (label, re.compile(pattern, re.IGNORECASE | re.UNICODE), desc)
            for label, pattern, desc in self.PATTERNS
        ]

    def extract(self, clause_id: int, clause: str) -> NERResult:
        entities: List[Entity] = []
        covered: List[Tuple[int, int]] = []

        for label, regex, desc in self._compiled:
            for m in regex.finditer(clause):
                start, end = m.start(), m.end()
                # Tránh overlap
                if any(s <= start < e or s < end <= e for s, e in covered):
                    continue
                entities.append(Entity(
                    text=m.group().strip(),
                    label=label,
                    start_char=start,
                    end_char=end,
                    description=desc,
                ))
                covered.append((start, end))

        # Sắp xếp theo vị trí xuất hiện
        entities.sort(key=lambda e: e.start_char)
        return NERResult(clause_id=clause_id, clause=clause, entities=entities)


# ─── ML-based NER (Production) ────────────────────────────────────────────────
class MLBasedNER:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.nlp = None
        self._load_model()

    def _load_model(self):
        try:
            import spacy
            if self.model_path and Path(self.model_path).exists():
                self.nlp = spacy.load(self.model_path)
                print(f"[NER] Loaded custom model from {self.model_path}")
            else:
                print("[NER] Custom model not found. Using rule-based fallback.")
                self.nlp = None
        except ImportError:
            print("[NER] spaCy not installed. Using rule-based fallback.")
            self.nlp = None

    def extract(self, clause_id: int, clause: str) -> Optional[NERResult]:
        if self.nlp is None:
            return None  # Trigger fallback
        doc = self.nlp(clause)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ENTITY_LABELS:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    description=ENTITY_LABELS.get(ent.label_, ""),
                ))
        return NERResult(clause_id=clause_id, clause=clause, entities=entities)


# ─── NER Pipeline ─────────────────────────────────────────────────────────────
class NERPipeline:
    def __init__(self, model_path: Optional[str] = None):
        self.ml_ner = MLBasedNER(model_path)
        self.rule_ner = RuleBasedNER()

    def process_clause(self, clause_id: int, clause: str) -> NERResult:
        result = self.ml_ner.extract(clause_id, clause)
        if result is not None:
            return result
        return self.rule_ner.extract(clause_id, clause)

    def process_file(self, input_path: str, output_path: str):
        input_file = Path(input_path)
        if not input_file.exists():
            input_file = Path("clauses.txt")
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

        # Làm sạch tag 
        clauses = []
        for line in input_file.read_text(encoding="utf-8").splitlines():
            clean_line = line.strip()
            if "]" in clean_line:
                clean_line = clean_line.rsplit("]", 1)[-1].strip()
            if clean_line:
                clauses.append(clean_line)

        results = []
        for i, clause in enumerate(clauses, start=1):
            result = self.process_clause(i, clause)
            results.append(result.to_dict())

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        total_entities = sum(len(r["entities"]) for r in results)
        print(f"\n[NER] Done! {len(results)} clauses, {total_entities} entities found.")
        print(f"[NER] Output saved to: {output_path}")
        return results

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT  = "output/clauses.txt"
    OUTPUT = "output/ner_results.json"

    pipeline = NERPipeline(model_path=None) 
    results = pipeline.process_file(INPUT, OUTPUT)

    # In preview
    print("\n── Preview (first 3 clauses) ──")
    for r in results[:3]:
        print(f"\nClause {r['clause_id']}: {r['clause'][:60]}...")
        for e in r["entities"]:
            print(f"  [{e['label']:8s}] {e['text']}")