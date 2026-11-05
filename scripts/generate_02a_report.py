import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def format_md_table(headers, rows):
    """Tạo bảng Markdown chuẩn khung, không vỡ dòng"""
    lines = []
    lines.append("| " + " | ".join(map(str, headers)) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)

def main():
    csv_path = PROJECT_ROOT / "docs" / "analysis" / "02a_ho_cau_hoi.csv"
    if not csv_path.exists():
        print(f"❌ Không tìm thấy file {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 1. Thống kê Ma trận Họ x Split (Số lượng)
    family_split = pd.crosstab(df['family'], df['split'], margins=True, margins_name='Tổng')
    rows_num = []
    for idx, row in family_split.iterrows():
        rows_num.append([idx, row.get('public', 0), row.get('train', 0), row.get('Tổng', 0)])
    table_num_md = format_md_table(["Họ câu hỏi", "Public Test", "Train Data", "Tổng cộng"], rows_num)

    # 2. Thống kê Ma trận Họ x Split (Tỷ lệ %)
    family_split_pct = (pd.crosstab(df['family'], df['split'], normalize='columns') * 100).round(2)
    rows_pct = []
    for idx, row in family_split_pct.iterrows():
        rows_pct.append([idx, f"{row.get('public', 0):.2f}%", f"{row.get('train', 0):.2f}%"])
    table_pct_md = format_md_table(["Họ câu hỏi", "Public Test (%)", "Train Data (%)"], rows_pct)

    # 3. Thống kê Thuộc tính Ngôn ngữ
    lang_impact = []
    for attr in ['khau_ngu', 'go_sai', 'viet_tat', 'nhieu_ve', 'loc']:
        cnt = df[attr].sum()
        pct = (cnt / len(df)) * 100
        pub_cnt = df[df['split']=='public'][attr].sum()
        trn_cnt = df[df['split']=='train'][attr].sum()
        lang_impact.append(f"- **`{attr}`**: Tổng **{cnt} câu** ({pct:.1f}%) — (Public: {pub_cnt} câu | Train: {trn_cnt} câu)")

    # 4. Trích xuất 3 ví dụ cho từng họ
    examples_md = []
    for fam in sorted(df['family'].unique()):
        if fam == 'Tổng': continue
        sample_q = df[df['family'] == fam]['question'].head(3).tolist()
        examples_md.append(f"### Họ {fam}")
        for i, q in enumerate(sample_q, 1):
            examples_md.append(f"{i}. \"{q}\"")
        examples_md.append("")

    # 5. Xuất nội dung Markdown chuẩn
    md_content = f"""# Báo cáo Phân tích Họ câu hỏi và Dữ liệu (Việc 2a)

## 1. Bảng Ma trận Họ câu hỏi × Split (Số lượng & Tỉ lệ %)

### Số lượng câu hỏi theo họ:
{table_num_md}

### Tỉ lệ % theo họ trên từng tập dữ liệu:
{table_pct_md}

---

## 2. Thống kê Thuộc tính Ngôn ngữ trên Dữ liệu

Tổng số câu hỏi phân tích: **{len(df)} câu** (1,000 Public Test + 1,200 Train Data mẫu).
{chr(10).join(lang_impact)}

---

## 3. Họ câu hỏi có chênh lệch lớn nhất giữa Train và Public

- **Họ xuất hiện nhiều nhất ở cả 2 tập:** `Q01` (Chiếm ~57.7% Public và 63.75% Train).
- **Các họ câu hỏi có sự lệch tỷ trọng đáng chú ý:**
  - **`Q19` (Đặc thù ngành):** Ở Public chiếm **4.2%** (42 câu) nhưng ở Train mẫu chỉ chiếm **0.58%** (7 câu).
  - **`Q08` (Đếm theo ngưỡng):** Ở Public chiếm **8.7%** (87 câu) cao hơn hẳn ở Train (**5.67%**).
  - **`Q05` (Kỳ/Ngày cao/thấp nhất):** Tập Train có tỷ lệ **3.75%** cao hơn Public (**1.7%**).

---

## 4. 3 Ví dụ tiêu biểu cho mỗi Họ câu hỏi

{chr(10).join(examples_md)}
"""

    out_md = PROJECT_ROOT / "docs" / "analysis" / "02a_tong_hop.md"
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✅ Đã sửa xong bảng và định dạng Markdown đẹp tại:\n   {out_md.resolve()}")

if __name__ == "__main__":
    main()