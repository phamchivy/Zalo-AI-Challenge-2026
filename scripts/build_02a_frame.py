# import json
# import csv
# from pathlib import Path

# #  Hàm load danh sách câu hỏi từ JSONL
# def load_questions(jsonl_path, split_name):
#     questions = []
#     path = Path(jsonl_path)
#     if not path.exists():
#         print(f"Warning: Không tìm thấy file {jsonl_path}")
#         return questions
        
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             data = json.loads(line)
#             questions.append({
#                 'split': split_name,
#                 'qid': data.get('id', ''),
#                 'business_id': data.get('business_id', ''),
#                 'question': data.get('question', ''),
#                 'family': '',      # Bạn sẽ mở CSV ra điền tay hoặc gán nhãn
#                 'khau_ngu': 0,     # 0 hoặc 1
#                 'go_sai': 0,       # 0 hoặc 1
#                 'viet_tat': 0,     # 0 hoặc 1
#                 'nhieu_ve': 0,     # 0 hoặc 1
#                 'loc': 0,          # 0 hoặc 1
#                 'sys_intent': '',  # Đọc bổ sung từ log nếu có
#                 'sys_status': '',  # Đọc bổ sung từ log nếu có
#                 'ghi_chu': ''
#             })
#     return questions

# def main():
#     # Đường dẫn thư mục đầu ra
#     out_dir = Path("docs/analysis")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_csv = out_dir / "02a_ho_cau_hoi.csv"

#     all_questions = []

#     # 2. Đọc file public test (1000 câu)
#     public_path = Path("data/public_test/questions.jsonl")
#     all_questions.extend(load_questions(public_path, "public"))

#     # 3. Đọc file train (lấy khoảng 600 câu hoặc lấy hết nếu cần)
#     train_path = Path("data/train/questions.jsonl")
#     all_questions.extend(load_questions(train_path, "train"))

#     if not all_questions:
#         print("Không tìm thấy dữ liệu câu hỏi nào!")
#         return

#     # 4. Ghi ra file CSV kết quả
#     fieldnames = [
#         'split', 'qid', 'business_id', 'question', 'family',
#         'khau_ngu', 'go_sai', 'viet_tat', 'nhieu_ve', 'loc',
#         'sys_intent', 'sys_status', 'ghi_chu'
#     ]

#     with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(all_questions)

#     print(f"✅ Đã tạo xong file khung CSV tại: {out_csv.resolve()}")
#     print(f"📊 Tổng số câu hỏi đã trích xuất: {len(all_questions)} câu.")

# if __name__ == "__main__":
#     main()

import json
import csv
import re
from pathlib import Path

# Thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. HEURISTIC PHÂN LOẠI HỌ CÂU HỎI (Q01-Q20) ---
def classify_family(q_text):
    q = q_text.lower()
    
    # Q02: Trung bình
    if any(k in q for k in ["trung bình", "bình quân", "bq"]):
        return "Q02"
    # Q03: So sánh 2 kỳ
    if any(k in q for k in ["so với", "tăng hay giảm", "hơn hay kém", "so th", "thay đổi bao nhiêu %"]):
        return "Q03"
    # Q04: Chuỗi nhiều kỳ theo thứ tự
    if any(k in q for k in ["lần lượt", "tiếp theo", "3 tháng liền", "các tháng"]):
        return "Q04"
    # Q06 & Q05: Top-N / Cao nhất / Thấp nhất
    if any(k in q for k in ["cao nhất", "thấp nhất", "đắt nhất", "nhiều nhất", "bán chạy nhất", "ít nhất", "top"]):
        if any(k in q for k in ["mặt hàng", "sp", "sản phẩm", "khách", "thợ", "nhân viên", "món", "dòng máy"]):
            return "Q06"
        return "Q05"
    # Q07: Xếp hạng theo rulebook
    if any(k in q for k in ["giờ cao điểm", "khách trung thành", "khách thân thiết"]):
        return "Q07"
    # Q08: Đếm theo ngưỡng
    if any(k in q for k in ["trở lên", "ít nhất", "từ ", "lớn hơn"]):
        return "Q08"
    # Q09: Tỉ trọng / Tỉ lệ (%)
    if any(k in q for k in ["chiếm bao nhiêu %", "tỉ trọng", "tỷ lệ", "mấy %"]):
        return "Q09"
    # Q10: Huỷ / Trả hàng
    if any(k in q for k in ["huỷ", "hủy", "trả hàng", "bị huỷ"]):
        return "Q10"
    # Q11: Hiệu ứng đổi giá
    if any(k in q for k in ["tăng giá", "đổi giá", "lên giá"]):
        return "Q11"
    # Q12: Nhập hàng / Nhà cung cấp
    if any(k in q for k in ["nhập", "nhà cung cấp", "ncc"]):
        return "Q12"
    # Q13: Kho / Tồn / Hết hàng
    if any(k in q for k in ["tồn", "kho", "hết hàng", "đứt hàng"]):
        return "Q13"
    # Q14: Lô / Hạn dùng
    if any(k in q for k in ["hạn dùng", "hsd", "lô"]):
        return "Q14"
    # Q15: Công nợ tại ngày
    if any(k in q for k in ["nợ", "công nợ"]):
        return "Q15"
    # Q17: Lãi gộp / Lãi ròng
    if any(k in q for k in ["lãi gộp", "lãi ròng", "lợi nhuận"]):
        return "Q17"
    # Q18: Giao hàng nhiều đợt
    if any(k in q for k in ["giao", "đợt giao"]):
        return "Q18"
    # Q19: Đặc thù ngành (§6: sửa xe, điện thoại, bỉm, thuốc kê đơn...)
    if any(k in q for k in ["sửa xe", "thợ", "bảo hành", "máy cũ", "trả góp", "kê đơn", "vá ruột"]):
        return "Q19"
    # Q20: Không trả lời được / chủ quan
    if any(k in q for k in ["ngon nhất", "chăm chỉ nhất", "thân thiện nhất", "khó tính nhất", "uy tín nhất"]):
        return "Q20"
    
    return "Q01" # Mặc định: Giá trị một số đo trong kỳ (doanh thu, số lượng, số đơn)

# --- 2. PHÁT HIỆN THUỘC TÍNH NGÔN NGỮ (0/1) ---
def detect_attributes(q_text):
    q = q_text.lower()
    
    khau_ngu = 1 if any(k in q for k in ["rứa", "nhiêu z", "sốp", "bữa nay", "ha", "nghen", "hen", "nè", "zị", "hỉ", "cho em xin", "giùm", "jùm", "bữa qua"]) else 0
    go_sai = 1 if any(k in q for k in ["mổi", "tiềng", "dduowcj", "nhta", "bna", "đơng", "kủa", "dúpp", "chên"]) else 0
    viet_tat = 1 if any(k in q for k in ["dthu", "th5", "th ", "t4/", "t5/", "đh", "sp", "bnhiu", "nv", "t3/", "q3/", "q4/"]) else 0
    nhieu_ve = 1 if ("," in q_text or ";" in q_text or ("với" in q and "lần lượt" in q)) else 0
    loc = 1 if any(k in q for k in ["nhóm", "kênh", "ca sáng", "ca chiều", "loại", "đường", "hãng"]) else 0
    
    return khau_ngu, go_sai, viet_tat, nhieu_ve, loc

# --- 3. TỰ ĐỘNG ĐỌC LOG HỆ THỐNG AUDIT / DEV ---
def load_audit_map():
    audit_path = PROJECT_ROOT / "runs" / "audit_v23" / "audit.tsv"
    audit_map = {}
    if audit_path.exists():
        with open(audit_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                audit_map[row['qid']] = row.get('status', 'unknown')
    return audit_map

def load_log_info(split_name, qid):
    if split_name == "public":
        log_file = PROJECT_ROOT / "runs" / "v232_public" / "logs" / f"{qid}.json"
    else:
        log_file = PROJECT_ROOT / "runs" / "v232_dev" / "logs" / f"{qid}.json"
        
    intent, decision_type = "unknown", "unknown"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                intent = data.get('intent', 'unknown')
                decision = data.get('decision', {})
                decision_type = decision.get('type', 'unknown')
        except Exception:
            pass
    return intent, decision_type

# --- 4. HÀM NẠP VÀ GÁN NHÃN CÂU HỎI ---
def load_questions(jsonl_path, split_name, audit_map, sample_limit_per_shop=None):
    questions = []
    path = Path(jsonl_path)
    if not path.exists():
        print(f"⚠️ Warning: Không tìm thấy file {jsonl_path}")
        return questions

    shop_counts = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = data.get('id', '')
            biz_id = data.get('business_id', '')

            # Lấy mẫu train (~40 câu/shop) nếu có quy định
            if sample_limit_per_shop:
                count = shop_counts.get(biz_id, 0)
                if count >= sample_limit_per_shop:
                    continue
                shop_counts[biz_id] = count + 1

            q_text = data.get('question', '')
            family = classify_family(q_text)
            kn, gs, vt, nv, loc = detect_attributes(q_text)

            # Lấy trạng thái hệ thống từ log
            intent, decision_type = load_log_info(split_name, qid)
            if split_name == "public":
                sys_status = decision_type if decision_type == "abstain" else audit_map.get(qid, decision_type)
            else:
                sys_status = decision_type

            questions.append({
                'split': split_name,
                'qid': qid,
                'business_id': biz_id,
                'question': q_text,
                'family': family,
                'khau_ngu': kn,
                'go_sai': gs,
                'viet_tat': vt,
                'nhieu_ve': nv,
                'loc': loc,
                'sys_intent': intent,
                'sys_status': sys_status,
                'ghi_chu': ''
            })
    return questions

def main():
    out_dir = PROJECT_ROOT / "docs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "02a_ho_cau_hoi.csv"

    audit_map = load_audit_map()
    all_questions = []

    # 1. Nạp Public Test (1000 câu)
    public_path = PROJECT_ROOT / "data" / "public_test" / "questions.jsonl"
    all_questions.extend(load_questions(public_path, "public", audit_map))

    # 2. Nạp Train (Mẫu ~40 câu/shop = ~600 câu)
    train_path = PROJECT_ROOT / "data" / "train" / "questions.jsonl"
    all_questions.extend(load_questions(train_path, "train", audit_map, sample_limit_per_shop=40))

    if not all_questions:
        print("❌ Không tìm thấy dữ liệu câu hỏi nào!")
        return

    fieldnames = [
        'split', 'qid', 'business_id', 'question', 'family',
        'khau_ngu', 'go_sai', 'viet_tat', 'nhieu_ve', 'loc',
        'sys_intent', 'sys_status', 'ghi_chu'
    ]

    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_questions)

    print(f"✅ Đã tạo xong file khung CSV chuẩn tại: {out_csv.resolve()}")
    print(f"📊 Tổng số câu hỏi đã trích xuất & gán nhãn thô: {len(all_questions)} câu.")

if __name__ == "__main__":
    main()