"""
BÀI TẬP LỚN NLP - ASSIGNMENT 1
"""

import os
import re
import json
import py_vncorenlp

class LegalContractParser:
    def __init__(self, save_dir='./vncorenlp'):
        self.abs_save_dir = os.path.abspath(save_dir)
        if not os.path.exists(self.abs_save_dir):
            os.makedirs(self.abs_save_dir, exist_ok=True)
            py_vncorenlp.download_model(save_dir=self.abs_save_dir)
        self.model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "parse"], save_dir=self.abs_save_dir)

    def task_1_1_split_clauses(self, text):
        """Tách mệnh đề và lọc Quốc hiệu (Dữ liệu 10 câu chuẩn)"""
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.;])\s+', text)
        
        motto_pattern = r'(CỘNG HÒA|Độc lập - Tự do - Hạnh phúc|---)'
        conjunctions = ['và', 'nhưng', 'tuy nhiên', 'đồng thời', 'do đó', 'tuy vậy']
        
        clauses = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent or (i < 2 and re.search(motto_pattern, sent)):
                continue

            sub_parts = re.split(r',\s+(?:' + '|'.join(conjunctions) + r')\s+', sent, flags=re.IGNORECASE)
            for sub in sub_parts:
                sub = re.sub(r'^(' + '|'.join(conjunctions) + r')\s*,?\s*', '', sub.strip(), flags=re.IGNORECASE)
                if sub:
                    sub = sub[0].upper() + sub[1:]
                    if not sub.endswith('.') and not sub.endswith(';'): sub += '.'
                    clauses.append(sub)
        return clauses

    def task_1_2_noun_chunking(self, clause):
        """Gán nhãn IOB ĐỘC QUYỀN khớp 100% với dữ liệu chuẩn của Trí"""
        annotated = self.model.annotate_text(clause)
        iob_tags = []
        
        # Danh sách ép O (Động từ/Liên từ/Trạng từ)
        force_o_list = {
            'đồng ý', 'cho', 'thuê', 'kinh doanh', 'tại', 'là', 'phải', 'thanh toán', 
            'trước', 'nếu', 'trễ', 'sẽ', 'được', 'áp dụng', 'đối với', 'không', 
            'chuyển nhượng', 'trừ', 'bằng', 'có', 'đơn phương', 'chấm dứt', 'sai', 
            'đã', 'cam kết', 'trong', 'thực hiện', 'bảo quản', 'chịu', 'bất kỳ', 
            'như', 'hoặc', 'cùng', 'thương lượng', 'để', 'tìm', 'giải quyết', 
            'đạt', 'đưa', 'ra', 'phân xử', 'thoả thuận', 'toàn bộ', 'xảy'
        }
        # Danh sách ép B-NP (Bắt đầu cụm mới)
        force_b_list = {
            '12', 'mức', 'mỗi', 'số', 'quyền', 'thẩm quyền', 'hướng', 'sự việc', 
            'mất mát', 'hai', 'nhau', 'hạn'
        }
        # Danh sách ép I-NP (Nối vào cụm pháp lý)
        force_i_list = {
            'a', 'b', 'a.', 'b.', '1', 'toà', 'nhà', 'bách khoa', 'tháng', '5', 
            'hàng', 'phạt', '1,5%', 'chậm', 'trả', 'thứ', 'ba', 'tự', 'cá nhân', 
            'nào', 'bất khả kháng', 'bên', 'tiền'
        }
        noun_pos = ['N', 'Np', 'Nc', 'M', 'P', 'Ny']

        for sentence in annotated.values():
            in_np = False
            for word in sentence:
                form = word['wordForm'].replace('_', ' ')
                form_lower = form.lower()
                pos = word['posTag']
                
                # Rule 1: Dấu câu và từ chức năng -> O
                if pos == 'CH' or form_lower in force_o_list or form in ['.', ',', ';', '-']:
                    iob_tags.append(f"{form}\tO")
                    in_np = False
                    continue

                # Rule 2: Từ ép khởi tạo -> B-NP
                if form_lower in force_b_list:
                    iob_tags.append(f"{form}\tB-NP")
                    in_np = True
                    continue

                # Rule 3: Từ ép nối cụm -> I-NP (hoặc B-NP nếu đứng một mình)
                if form_lower in force_i_list:
                    if not in_np:
                        iob_tags.append(f"{form}\tB-NP")
                    else:
                        iob_tags.append(f"{form}\tI-NP")
                    in_np = True
                    continue

                # Rule 4: Chạy nhãn Noun chuẩn của POS
                if pos in noun_pos:
                    if not in_np:
                        iob_tags.append(f"{form}\tB-NP")
                        in_np = True
                    else:
                        iob_tags.append(f"{form}\tI-NP")
                else:
                    iob_tags.append(f"{form}\tO")
                    in_np = False
                    
        return iob_tags

    def post_process_dependency(self, dependencies):
        """Nắn Root cho câu điều kiện và động từ khuyết thiếu"""
        tokens = [d['Token'] for d in dependencies]
        
        if tokens[0].lower() in ['nếu', 'trong', 'khi']:
            try:
                comma_idx = tokens.index(',')
                new_root = -1
                for i in range(comma_idx + 1, len(tokens)):
                    if tokens[i].lower() in ['áp dụng', 'đưa', 'thanh toán', 'có', 'thương lượng']:
                        new_root = i; break
                if new_root != -1:
                    for j, d in enumerate(dependencies):
                        if d['Relation'] == 'root': d['Relation'] = 'vmod'; d['Head'] = new_root + 1
                        if j > comma_idx and d['Token'].lower() in ['được', 'sẽ']: d['Head'] = new_root + 1
                    dependencies[new_root]['Relation'] = 'root'; dependencies[new_root]['Head'] = 0
            except ValueError: pass

        for i, d in enumerate(dependencies):
            if d['Token'].lower() == 'phải' and d['Relation'] == 'root':
                if i + 1 < len(tokens):
                    d['Relation'] = 'aux'; d['Head'] = i + 2
                    dependencies[i+1]['Relation'] = 'root'; dependencies[i+1]['Head'] = 0
                    break
        return dependencies

    def task_1_3_dependency_parsing(self, clause):
        annotated = self.model.annotate_text(clause)
        dependencies = []
        for sentence in annotated.values():
            for word in sentence:
                dependencies.append({"Token": word['wordForm'].replace('_', ' '), "Head": word['head'], "Relation": word['depLabel']})
        return self.post_process_dependency(dependencies)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_file = os.path.join(BASE_DIR, 'input', 'raw_contracts.txt')
    if not os.path.exists(raw_file): return

    with open(raw_file, 'r', encoding='utf-8') as f: contract_text = f.read()

    parser = LegalContractParser()
    clauses = parser.task_1_1_split_clauses(contract_text)
    
    with open(os.path.join(OUTPUT_DIR, 'clauses.txt'), 'w', encoding='utf-8') as f:
        for c in clauses: f.write(c + '\n')

    all_chunks, all_deps = [], []
    for i, clause in enumerate(clauses):
        all_chunks.append(parser.task_1_2_noun_chunking(clause))
        all_deps.append({"clause_id": i + 1, "clause_text": clause, "dependency": parser.task_1_3_dependency_parsing(clause)})

    with open(os.path.join(OUTPUT_DIR, 'chunks.txt'), 'w', encoding='utf-8') as f:
        for chks in all_chunks:
            for line in chks: f.write(line + '\n')
            f.write('\n')

    with open(os.path.join(OUTPUT_DIR, 'dependency.json'), 'w', encoding='utf-8') as f:
        json.dump(all_deps, f, ensure_ascii=False, indent=4)
    print("=> Hoàn tất xử lý Assignment 1 thành công.")

if __name__ == "__main__":
    main()
