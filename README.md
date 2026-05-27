# AIDEOM-VN — Phát triển kinh tế Việt Nam trong kỉ nguyên AI

Bài tập lớn học phần **Các mô hình ra quyết định**: giải 12 bài toán chính sách kinh tế số của Việt Nam bằng các mô hình tối ưu hóa và học tăng cường, trên dữ liệu thực tế 2020–2025. Toàn bộ kết quả được tính bằng Python và tái lập được (reproducible), kèm hai dashboard trực quan (web tĩnh và Streamlit) và báo cáo Word.

| | |
|---|---|
| **Sinh viên** | Nguyễn Bảo Khánh |
| **Mã sinh viên** | 23051266 |
| **Học phần** | Các mô hình ra quyết định |
| **Giảng viên hướng dẫn** | TS. Phạm Văn Khánh |
| **Năm học** | 2025 – 2026 |

## Cấu trúc thư mục

```
.
├── app.py                                          # Dashboard Streamlit (12 bài, tương tác)
├── requirements.txt                                # Danh sách thư viện Python
├── README.md                                       # Tệp này
├── Phát_triển_kinh_tế_Việt_Nam_trong_kỉ_nguyên_AI.html   # Dashboard web tĩnh (mở bằng trình duyệt)
└── Phát_triển_kinh_tế_Việt_Nam_trong_kỉ_nguyên_AI.docx   # Báo cáo đầy đủ (Word)
```

## Yêu cầu hệ thống

- Python 3.10 – 3.12
- Kết nối Internet khi mở dashboard web tĩnh (`.html`) để tải Chart.js và KaTeX từ CDN. Nếu không có mạng, nội dung và số liệu vẫn hiển thị nhưng biểu đồ/công thức sẽ không render.

## Cài đặt

```bash
# (khuyến nghị) tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# cài thư viện
pip install -r requirements.txt
```

## Cách chạy

### 1. Dashboard Streamlit (tương tác đầy đủ)

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`. Chọn từng bài ở thanh bên trái; các thanh trượt (ngân sách, hệ số công bằng λ, chiết khấu ρ, trọng số AI, số episodes…) cho phép chạy lại mô hình theo thời gian thực.

> Mọi con số trong dashboard được **tính trực tiếp** bằng `numpy / scipy / PuLP / pymoo` mỗi khi mở trang — không hard-code, nên hoàn toàn tái lập được.

### 2. Dashboard web tĩnh

Mở trực tiếp tệp `Phát_triển_kinh_tế_Việt_Nam_trong_kỉ_nguyên_AI.html` bằng trình duyệt (Chrome, Edge, Firefox…). Không cần cài đặt gì thêm. Mỗi bài có mô hình toán (KaTeX), mã Python, biểu đồ (Chart.js) và phần thảo luận chính sách.

### 3. Báo cáo

Mở `Phát_triển_kinh_tế_Việt_Nam_trong_kỉ_nguyên_AI.docx` bằng Microsoft Word hoặc LibreOffice. Báo cáo có mục lục tự động, công thức, bảng kết quả và diễn giải chính sách cho cả 12 bài.

## Nội dung 12 bài tập

| Cấp độ | Bài | Chủ đề | Phương pháp / công cụ |
|--------|-----|--------|------------------------|
| Dễ | 1 | Hàm sản xuất Cobb–Douglas mở rộng | TFP, phân rã tăng trưởng · numpy |
| Dễ | 2 | Phân bổ ngân sách 4 hạng mục | LP + shadow price · scipy/PuLP |
| Dễ | 3 | Chỉ số ưu tiên ngành | Chuẩn hóa min-max + trọng số |
| Trung bình | 4 | LP phân bổ ngân sách theo vùng | LP + ràng buộc công bằng · PuLP |
| Trung bình | 5 | Chọn danh mục dự án | MIP / knapsack nhị phân · PuLP |
| Trung bình | 6 | Xếp hạng vùng ưu tiên AI | TOPSIS + Entropy |
| Khá khó | 7 | Tối ưu đa mục tiêu | NSGA-II (4 mục tiêu) · pymoo |
| Khá khó | 8 | Tối ưu động 2026–2035 | Điều khiển tối ưu · scipy SLSQP |
| Khá khó | 9 | Tác động AI tới lao động | LP tối đa việc làm ròng (NetJob) |
| Khó | 10 | Quy hoạch ngẫu nhiên 2 giai đoạn | Stochastic LP (VSS, EVPI) · PuLP |
| Khó | 11 | Chính sách thích nghi | Q-learning (MDP) · numpy |
| Khó | 12 | Hệ thống tích hợp 6 module | Tích hợp Bài 1–11 |

### Một số phát hiện chính

- **Bài 1:** TFP và số hóa là động lực tăng trưởng chính (đóng góp ~49% và ~10%).
- **Bài 4:** ràng buộc công bằng λ = 0,70 (theo đề) là **vô nghiệm** (λ khả thi tối đa ≈ 0,683); ở mức λ = 0,65 chi phí công bằng đo được ~16,8% GDP gain.
- **Bài 8:** quỹ đạo đầu tư tối ưu mang tính **front-loaded** (đầu tư sớm để vốn và TFP cộng dồn).
- **Bài 9:** với cơ cấu hệ số hiện tại, AI **không** gây mất việc ròng nếu đi kèm đào tạo lại (tổng NetJob ~1,37 triệu việc).
- **Bài 11:** chính sách thích nghi π* (Q-learning) **vượt** mọi luật cố định nhờ sắp xếp số hóa → AI → bao trùm đúng thời điểm.

## Nguồn dữ liệu

- Cục Thống kê quốc gia (NSO/GSO)
- Ngân hàng Thế giới (World Bank)
- Bộ Khoa học – Công nghệ (MoST)
- Global Innovation Index 2025 (WIPO)

Số liệu được làm tròn phục vụ giảng dạy. Khung chính sách tham chiếu: Nghị quyết 57-NQ/TW; các Quyết định 749, 127, 411/QĐ-TTg; cam kết COP26.

## Công nghệ sử dụng

- **Tính toán:** numpy, pandas, scipy
- **Tối ưu hóa:** PuLP (LP/MIP, solver CBC), pymoo (NSGA-II)
- **Dashboard:** Streamlit (`app.py`); HTML/JavaScript với Chart.js và KaTeX (dashboard web tĩnh)

## Ghi chú

Dự án có sử dụng công cụ AI hỗ trợ trong quá trình lập trình và trình bày, theo đúng quy định cho phép của học phần. Mọi mô hình toán, lựa chọn tham số và diễn giải kết quả đều được kiểm tra thủ công để bảo đảm tính đúng đắn và tái lập.
