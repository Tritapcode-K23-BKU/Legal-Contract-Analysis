"""
SRL FINAL VERSION (fully fixed & cleaned)
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


# ─── SCHEMA ─────────────────────────────────────────────
LEGAL_PREDICATES = {
    "thanh toán": "payment obligation",
    "bàn giao": "handover",
    "đơn phương chấm dứt": "termination",
    "chấm dứt": "termination",
    "chuyển nhượng": "assignment",
    "bảo quản": "preservation",
    "thuê": "lease",
    "thương lượng": "negotiation",
    "phân xử": "adjudication",
    "áp dụng": "application",
    "đồng ý": "consent",
    "chịu trách nhiệm": "responsibility",
    "đưa ra": "submission",
    "là": "copula",
}


VERB_PHRASES = [
    "đơn phương chấm dứt",
    "thanh toán toàn bộ",
    "chịu trách nhiệm",
    "áp dụng",
    "chuyển nhượng",
    "thanh toán",
    "bảo quản",
    "thương lượng",
    "phân xử",
    "đồng ý",
    "đưa ra",
    "là",
    "thuê"
]


# ─── DATA ───────────────────────────────────────────────
@dataclass
class SRLFrame:
    predicate: str
    predicate_type: str
    roles: Dict[str, str] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self):
        return {
            "predicate": self.predicate,
            "predicate_type": self.predicate_type,
            "roles": self.roles,
            "explanation": self.explanation,
        }


@dataclass
class SRLResult:
    clause_id: int
    clause: str
    frames: List[SRLFrame]

    def to_dict(self):
        return {
            "clause_id": self.clause_id,
            "clause": self.clause,
            "frames": [f.to_dict() for f in self.frames],
        }


# ─── SRL CORE ───────────────────────────────────────────
class RuleBasedSRL:

    # ---------- BASIC ----------
    def _agent(self, text):
        m = re.search(r"\b(Bên\s+[A-Z]|hai\s+bên)\b", text)
        return m.group(1) if m else None

    def _condition(self, text):
        m = re.search(r"(?i)(Nếu|Trong trường hợp|Trong quá trình|trừ khi|khi)\s+(.+?)(?:[,;]|$)", text)
        if m:
            cond = m.group(2).strip().rstrip(".")
            main = text.replace(m.group(0), "").strip()
            return cond, main
        return None, text

    def _time(self, text):
        m = re.search(r"(?i)(trước ngày \d+ hàng tháng|\d+ (?:ngày|tháng|năm))", text)
        return m.group(1) if m else None

    def _recipient(self, text):
        m = re.search(r"(?i)(?:cho|đối với)\s+(Bên\s+[A-Z]|bên\s+thứ\s+ba)\b", text)
        return m.group(1) if m else None

    def _location(self, text):
        # ❗ Đã chặn không cho location nuốt cụm "để..."
        m = re.search(r"(?i)(?:tại|ở|trong|ra)\s+((?:tầng\s+\d+\s+)?(?:tòa\s+nhà|mặt\s+bằng|văn\s+phòng|Tòa\s+án)[^,;.]*)", text)
        if m:
            loc = m.group(1).strip()
            loc = re.split(r"(?i)\s+để\b", loc)[0].strip()
            return loc
        return None

    def _purpose(self, text):
        # ❗ Đã bỏ logic chặn "phân xử" thủ công để code xử lý tự nhiên
        m = re.search(r"(?i)để\s+([^,;.]+)", text)
        return m.group(1).strip() if m else None

    def _neg(self, text):
        return bool(re.search(r"(?i)\bkhông\b", text))

    # ---------- SMART PREDICATE FILTER ----------
    def _predicates(self, text):
        lower = text.lower()
        found = [vp for vp in VERB_PHRASES if vp in lower]

        # remove nested
        clean = []
        for p in found:
            if not any(p != q and p in q for q in found):
                clean.append(p)

        if "thuê" in clean:
            if not re.search(r"(?i)\bthuê\s+(mặt bằng|nhà|tài sản)\b", text):
                clean.remove("thuê")

        if "là" in clean:
            return ["là"]

        return clean or ["không xác định"]

    # ---------- THEME ----------
    def _theme(self, text, predicate, agent, time_, recipient, location, purpose):
        idx = text.lower().find(predicate)
        if idx == -1:
            return None

        after = text[idx + len(predicate):]

        after = re.split(r"(?i)\b(nếu|khi|trừ khi|trong trường hợp|trong quá trình)\b", after)[0]

        if time_:
            after = after.replace(time_, "")
        if location:
            after = after.replace(location, "")
        if purpose:
            after = after.replace(f"để {purpose}", "")

        after = re.sub(r"(?i)\bcho\s+(Bên\s+[A-Z]|bên\s+thứ\s+ba)\b", "", after)

        if agent:
            after = after.replace(agent, "")

        after = re.sub(r"(?i)\bđối với\s+", "", after)
        after = re.sub(r"(?i)^(cho|tại|trong|về|vào)\s+", "", after)

        after = re.split(r"[,;]", after)[0]
        after = after.strip(" ,;.")

        after = re.sub(r"(?i)\b(tại|ở|trong|ra)\s*$", "", after).strip()

        return after if len(after) > 1 else None

    # ---------- EXPLAIN ----------
    def _explain(self, f: SRLFrame):
        r = f.roles
        parts = []

        if "Agent" in r:
            parts.append(f"{r['Agent']} thực hiện '{f.predicate}'")
        else:
            parts.append(f"Hành động '{f.predicate}'")

        if "Theme" in r:
            parts.append(f"đối với '{r['Theme']}'")

        if "Recipient" in r:
            parts.append(f"hướng đến '{r['Recipient']}'")

        if "Condition" in r:
            parts.append(f"khi '{r['Condition']}'")

        if "Time" in r:
            parts.append(f"vào '{r['Time']}'")

        if "Purpose" in r:
            parts.append(f"nhằm '{r['Purpose']}'")

        if "Location" in r:
            parts.append(f"tại '{r['Location']}'")

        if "Negation" in r:
            parts.append("(phủ định)")

        return "; ".join(parts) + "."

    # ---------- MAIN ----------
    def analyze(self, cid, clause):
        frames = []

        agent = self._agent(clause)
        condition, main = self._condition(clause)
        preds = self._predicates(main)

        for p in preds:
            roles = {}

            if agent:
                roles["Agent"] = agent
            if condition:
                roles["Condition"] = condition

            if p == "là":
                t = self._time(clause)
                if t:
                    roles["Time"] = t
                frames.append(SRLFrame("là", "copula", roles, "Mệnh đề định nghĩa/thời hạn."))
                continue

            t = self._time(main)
            rec = self._recipient(main)
            loc = self._location(main)
            pur = self._purpose(main)

            if t:
                roles["Time"] = t
            if rec:
                roles["Recipient"] = rec
            if loc:
                roles["Location"] = loc
            if pur:
                roles["Purpose"] = pur
            if self._neg(main):
                roles["Negation"] = "không"

            if p == "thuê" and "cho" in clause and rec:
                roles["Agent"] = rec
                roles.pop("Recipient", None)

            if p == "áp dụng":
                roles.pop("Agent", None)

            # ❗ CÚ FIX LỊCH SỬ CHO CÂU BỊ ĐỘNG (Câu 4, Câu 10)
            # (.*?) sẽ lấy trọn vẹn "mức phạt 1,5% mỗi ngày" mà không sợ dấu phẩy
            # (?:\s+sẽ)? sẽ gom chữ "sẽ" vào regex để không lọt ra Theme
            passive_match = re.search(r"(?i)(.*?)(?:\s+sẽ)?\s+(?:được|bị)\s+" + re.escape(p), main)
            
            if passive_match:
                theme = passive_match.group(1).strip()
            else:
                theme = self._theme(main, p, roles.get("Agent"), t, rec, loc, pur)

            if theme:
                roles["Theme"] = theme

            frame = SRLFrame(p, LEGAL_PREDICATES.get(p, "legal action"), roles)
            frame.explanation = self._explain(frame)

            frames.append(frame)

        return SRLResult(cid, clause, frames)


# ─── PIPELINE ───────────────────────────────────────────
# ─── PIPELINE ───────────────────────────────────────────
class SRLPipeline:

    def __init__(self):
        self.srl = RuleBasedSRL()

    def process_file(self, inp, out):
        lines = Path(inp).read_text(encoding="utf-8").splitlines()

        clauses = []
        for l in lines:
            if "]" in l: l = l.split("]", 1)[-1]
            l = l.strip()
            if l: clauses.append(l)

        results = []
        for i, c in enumerate(clauses, 1):
            results.append(self.srl.analyze(i, c).to_dict())

        Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Done SRL:", len(results))
        
        # ❗ THÊM DÒNG NÀY ĐỂ MAIN.PY KHÔNG BỊ LỖI NONETYPE
        return results 


# ─── RUN ────────────────────────────────────────────────
if __name__ == "__main__":
    SRLPipeline().process_file("input/clauses.txt", "output/srl_results.json")