# AIDEOM-VN · Analytics Terminal

Web trình bày bộ **12 bài tập Mô hình ra quyết định** — phát triển kinh tế Việt
Nam trong kỉ nguyên AI, sử dụng dữ liệu thực giai đoạn 2020–2025. Giao diện theo
phong cách **command-center tối** với accent neon: mục lục **sidebar bên trái**,
lưới nền kỹ thuật, phông Space Grotesk / JetBrains Mono. Mặc định nền tối, có nút
☀/☾ để chuyển sáng/tối.

Toàn bộ phép tính và biểu đồ được tính **trực tiếp trong trình duyệt** bằng
JavaScript — không cần máy chủ Python để xử lý. File `app.py` chỉ là một cách tùy
chọn để phục vụ trang qua HTTP.

---

## 1. Yêu cầu trước khi chạy

- **Trình duyệt** hiện đại (Chrome, Edge, Firefox, Safari).
- **Kết nối Internet** — để tải thư viện vẽ biểu đồ (Chart.js), công thức toán
  (KaTeX) và phông chữ từ CDN. Nếu không có mạng, trang vẫn hiển thị đầy đủ chữ và
  số liệu, nhưng biểu đồ sẽ trống (có banner cảnh báo).
- **Python 3** (chỉ cần nếu dùng Cách 2 hoặc Cách 3 bên dưới).

---

## 2. Cách chạy web

### Cách 1 — Máy chủ tĩnh Python (khuyến nghị, không cần cài gì)

```bash
cd aideom_terminal
python -m http.server 5000
```

Sau đó mở trình duyệt: **http://127.0.0.1:5000**

> Đây là cách ổn định nhất. Mở qua `http://` giúp trình duyệt tải đủ thư viện và
> tránh lỗi "Script error." hay gặp khi mở file trực tiếp.

### Cách 2 — Dùng Flask (app.py)

```bash
cd aideom_terminal
pip install -r requirements.txt
python app.py
```

Mở: **http://127.0.0.1:5000** (Ctrl+C để dừng).

### Cách 3 — Mở trực tiếp (nhanh nhưng kém ổn định)

Nhấp đúp vào `index.html`. Cách này đôi khi bị trình duyệt chặn tải thư viện
(do giao thức `file://`), khiến biểu đồ trống. Nếu gặp lỗi, hãy dùng Cách 1.

### Đưa lên mạng (deploy)

Trang là một file tĩnh nên triển khai rất đơn giản:

- **Vercel / Netlify / GitHub Pages:** chỉ cần upload `index.html` (không cần
  `app.py`). Trên Vercel có thể kéo-thả thư mục hoặc đặt `index.html` vào `public/`.

---

## 3. Nội dung web gồm những gì

Trang chủ + 12 bài tập, chia theo 4 cấp độ khó tăng dần:

| Cấp độ | Bài | Nội dung |
|--------|-----|----------|
| **Dễ** | 1 | Hàm sản xuất Cobb-Douglas mở rộng — TFP, MAPE, phân rã tăng trưởng, kịch bản 2030 (có thanh trượt mô phỏng) |
| | 2 | Phân bổ ngân sách 4 hạng mục (LP) — giá đối ngẫu, độ nhạy ngân sách |
| | 3 | Chỉ số ưu tiên ngành — chuẩn hóa min-max, xếp hạng 10 ngành, độ nhạy trọng số |
| **Trung bình** | 4 | LP phân bổ ngân sách số theo vùng — ràng buộc công bằng vùng miền |
| | 5 | MIP chọn dự án chuyển đổi số — knapsack nhị phân, ràng buộc tiên quyết |
| | 6 | TOPSIS xếp hạng 6 vùng — trọng số chuyên gia vs Entropy, AHP |
| **Khá khó** | 7 | NSGA-II đa mục tiêu — đường biên Pareto, tọa độ song song, nghiệm thỏa hiệp |
| | 8 | Tối ưu động 2026–2035 — quỹ đạo vốn, phân tích cú sốc |
| | 9 | Tác động AI tới lao động — NetJob ròng, ngưỡng đào tạo, trần an sinh |
| **Khó** | 10 | Quy hoạch ngẫu nhiên 2 giai đoạn — VSS, EVPI, robust optimization |
| | 11 | Q-learning chính sách thích nghi — MDP, π*, so sánh rule-based |
| | 12 | AIDEOM-VN tích hợp — 6 module, 5 kịch bản chính sách (chia 4 tab) |

Mỗi bài đều có: **mục tiêu học tập**, **mô hình toán** (công thức), **các câu kết
quả** (bảng số + biểu đồ tương tác), và **thảo luận chính sách**.

---

## 4. Cách trình bày & điều hướng

- **Trang chủ** (`#home`): banner tổng quan, 4 chỉ số vĩ mô 2025, ma trận độ khó,
  danh mục 12 module (nhấp để mở).
- **Sidebar bên trái**: mục lục 12 bài + thông tin sinh viên ở cuối.
- **Liên kết trực tiếp** tới từng bài qua hash trên URL:
  - Trang chủ: `index.html#home`
  - Từng bài: `index.html#bai1`, `#bai2`, … `#bai12`
- **Nút ☀/☾** ở góc phải trên: chuyển nền sáng/tối (tự ghi nhớ lựa chọn).

---

## 5. Cấu trúc thư mục

```
aideom_terminal/
├── index.html        # Toàn bộ web: HTML + CSS + JS + dữ liệu (tự chứa)
├── app.py            # Máy chủ Flask tùy chọn (Cách 2)
├── requirements.txt  # Thư viện cần cho app.py (chỉ có flask)
└── README.md         # File hướng dẫn này
```

**Bên trong `index.html`** (một file duy nhất, tự chứa):

- `<style>` — toàn bộ CSS giao diện (chủ đề tối/sáng, sidebar, thẻ, bảng, biểu đồ).
- Phần HTML — header, sidebar mục lục, khung 12 trang bài.
- `<script>` — chứa:
  - **Đối tượng `R`**: dữ liệu kết quả đã tính sẵn cho cả 12 bài.
  - **Hàm `renderBaiN()` / `initBaiN()`**: dựng nội dung và vẽ biểu đồ từng bài.
  - **Bộ định tuyến (router)**: hiển thị bài theo hash URL, đổi chủ đề sáng/tối,
    bắt lỗi và cảnh báo nếu thư viện CDN không tải được.

---

## 6. Xử lý sự cố

| Hiện tượng | Cách khắc phục |
|------------|----------------|
| Biểu đồ trống / banner đỏ cảnh báo | Kiểm tra mạng Internet; dùng Cách 1 thay vì mở file trực tiếp |
| `Uncaught Error: Script error.` | Mở qua `http://` (Cách 1), không nhấp đúp file |
| Cổng 5000 đang bận | Đổi cổng: `python -m http.server 5050` rồi mở `http://127.0.0.1:5050` |
| Muốn xem lỗi chi tiết | Mở F12 → tab **Console**, chụp dòng đỏ |

---

## 7. Nguồn dữ liệu

Cục Thống kê quốc gia (NSO/GSO), Ngân hàng Thế giới, Bộ Khoa học - Công nghệ,
Global Innovation Index 2025. Số liệu CSV được làm tròn phục vụ giảng dạy; khi
viết luận văn cần truy xuất số gốc tại `nso.gov.vn`.
