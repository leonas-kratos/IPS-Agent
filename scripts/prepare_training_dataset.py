#!/usr/bin/env python3
"""
scripts/prepare_training_dataset.py
===================================
Tự động nạp, phân tích và chuyển đổi toàn bộ tri thức trong repository IPS thành:
1. Tập dữ liệu huấn luyện (Training Dataset) chuyên sâu định dạng JSONL (Chat) và JSON (Alpaca).
2. File cấu hình Ollama Modelfile chứa tri thức nén toán học, phần cứng, bài báo và mã nguồn xe CAR.
3. Cập nhật Master SKILL.md theo chuẩn Antigravity.
"""

import os
import re
import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT_DIR / "Paper"
CAR_DIR = ROOT_DIR / "System" / "CAR"
DOC_DIR = ROOT_DIR / "Document"
SKILLS_DIR = ROOT_DIR / "skills"
DATASET_DIR = ROOT_DIR / "dataset"

SYSTEM_PROMPT = """Bạn là Kỹ sư Nghiên cứu Cấp cao & Kiến trúc sư Hệ thống Định vị Trong nhà (Indoor Positioning System - IPS) UWB.
Bạn nắm vững 100% mã nguồn hệ thống xe tự hành CAR, các bài báo nghiên cứu quốc tế & nội bộ phòng lab, tài liệu phần cứng Decawave DWM1001C và các giải thuật toán học (LSE, EKF, UKF, FGO, CIR Deep Learning).
Khi trả lời, bạn luôn tuân thủ chuẩn Antigravity:
1. Trình bày Markdown chuyên nghiệp, kẻ bảng biểu chi tiết có phân loại và số liệu rõ ràng.
2. Trích dẫn đích danh tên bài báo, năm và số liệu định lượng sai số thực nghiệm (RMSE, cm, mm, %).
3. Cung cấp công thức toán học tường minh (LaTeX) và đoạn mã Python/C++ thực thi cụ thể khi được yêu cầu.
4. Đưa ra phân tích kỹ thuật sắc sảo và giải pháp khả thi cho hệ thống thực nghiệm của phòng lab."""

def get_text_from_pdf(pdf_path: Path, max_pages: int = 6) -> str:
    try:
        cmd = ["pdftotext", "-f", "1", "-l", str(max_pages), str(pdf_path), "-"]
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

def scan_all_papers():
    papers = []
    pdfs = sorted(list(PAPER_DIR.rglob("*.pdf")))
    for p in pdfs:
        txt = get_text_from_pdf(p, max_pages=6)
        if not txt:
            continue
        
        title = p.stem.replace("_", " ")
        is_our = "OurPapers" in str(p)
        
        # Metadata
        doi_m = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', txt[:2000])
        doi = doi_m.group(0).rstrip('.,') if doi_m else ("Báo cáo Nội bộ Lab" if is_our else "N/A")
        
        years = re.findall(r'\b(201\d|202\d)\b', txt[:2000])
        year = years[0] if years else "2023+"
        
        # Abstract
        abs_match = re.search(r'abstract[:\s\n]+(.*?)(?=(?:index terms|keywords|introduction|i\. introduction))', txt, re.IGNORECASE | re.DOTALL)
        abstract = abs_match.group(1).strip() if abs_match else txt[:1000].strip()
        abstract = re.sub(r'\s+', ' ', abstract)[:1200]
        
        # Categorize
        txt_l = txt.lower()
        if any(k in txt_l for k in ["gramian", "cnn", "deep learning", "neural", "lstm", "mlp"]):
            cat = "Deep Learning & AI"
        elif any(k in txt_l for k in ["ekf", "ukf", "kalman", "csrukf", "rts"]):
            cat = "Bộ lọc Kalman / EKF / UKF"
        elif any(k in txt_l for k in ["imu", "pdr", "ins", "fusion", "mimu"]):
            cat = "Sensor Fusion (IMU/PDR/LiDAR)"
        else:
            cat = "Hình học & Bản đồ kênh"
            
        # Metric
        metrics = re.findall(r'([^\n.]{0,60}?(?:rmse|positioning error|ranging error|accuracy)[^\n.]{0,40}?\b\d+\.?\d*\s*(?:m|cm|mm|%)[^\n.]{0,40})', txt, re.IGNORECASE)
        best_metric = re.sub(r'\s+', ' ', metrics[0].strip()) if metrics else "Giảm thiểu sai số NLOS và cải thiện độ ổn định"
        
        papers.append({
            "path": str(p.relative_to(ROOT_DIR)),
            "filename": p.name,
            "title": title,
            "is_our": is_our,
            "doi": doi,
            "year": year,
            "category": cat,
            "abstract": abstract,
            "best_metric": best_metric
        })
    return papers

def scan_car_code():
    code_files = []
    if CAR_DIR.exists():
        for py in sorted(CAR_DIR.glob("*.py")):
            try:
                content = py.read_text(encoding="utf-8", errors="ignore")
                code_files.append({
                    "filename": py.name,
                    "rel_path": str(py.relative_to(ROOT_DIR)),
                    "content": content[:4000]
                })
            except Exception:
                pass
    return code_files

def build_training_dataset(papers, code_files):
    examples = []
    
    # =========================================================================
    # PHẦN 1: BỘ CÂU HỎI VỀ TỪNG BÀI BÁO NGHIÊN CỨU (48+ BÀI)
    # =========================================================================
    for p in papers:
        # Tóm tắt & phương pháp
        q1 = f"Hãy tóm tắt bài báo '{p['title']}', nêu rõ phương pháp và trường phái nghiên cứu."
        a1 = f"""Bài báo **{p['title']}** ({p['year']}) thuộc trường phái **{p['category']}**.

**Mã DOI / Xuất bản:** `{p['doi']}`
**Phân loại:** {'Báo cáo Lab Tác Giả (OurPapers)' if p['is_our'] else 'Công bố Quốc tế'}

### 1. Đóng góp & Phương pháp cốt lõi:
{p['abstract']}

### 2. Sai số & Hiệu suất thực nghiệm đạt được:
- **Chỉ số sai số / Độ chính xác:** **{p['best_metric']}**.
- **Khả năng ứng dụng:** Tối ưu hóa ước lượng tọa độ trong điều kiện che khuất phi đường truyền thẳng (NLOS) và phản xạ đa đường (Multipath)."""
        examples.append({"instruction": q1, "input": "", "output": a1})

        # Câu hỏi về hiệu năng & số liệu
        q2 = f"Sai số tốt nhất (Best Error / RMSE) đạt được của bài báo '{p['title']}' là bao nhiêu?"
        a2 = f"Trong bài báo **{p['title']}**, hiệu năng định lượng tốt nhất đạt được là: **{p['best_metric']}**. Phương pháp cốt lõi thuộc nhóm **{p['category']}**."
        examples.append({"instruction": q2, "input": "", "output": a2})

    # =========================================================================
    # PHẦN 2: BẢNG TỔNG HỢP DANH MỤC VÀ SO SÁNH THEO TRƯỜNG PHÁI
    # =========================================================================
    q_all = "Liệt kê toàn bộ các bài báo trong kho dữ liệu kèm trường phái và sai số tốt nhất."
    table_lines = ["| STT | Tên bài báo | Năm | Trường phái | Sai số tốt nhất |",
                   "|:---:|:---|:---:|:---|:---|"]
    for idx, p in enumerate(papers, 1):
        table_lines.append(f"| {idx} | **{p['title']}** | {p['year']} | {p['category']} | {p['best_metric']} |")
    a_all = f"## 📚 Bảng Tổng Hợp Toàn Bộ {len(papers)} Bài Báo Trong Kho Nghiên Cứu UWB\n\n" + "\n".join(table_lines)
    examples.append({"instruction": q_all, "input": "", "output": a_all})

    # =========================================================================
    # PHẦN 3: CÁC NGHIÊN CỨU CỦA CHÍNH PHÒNG LAB (OURPAPERS)
    # =========================================================================
    q_our = "Cho biết các bài báo nghiên cứu của chính phòng lab (OurPapers) và đóng góp chính."
    our_papers = [p for p in papers if p['is_our']]
    a_our = f"## 🔬 Các Nghiên Cứu Của Phòng Lab (OurPapers - {len(our_papers)} bài báo)\n\n"
    for idx, p in enumerate(our_papers, 1):
        a_our += f"### {idx}. {p['title']}\n- **Tệp nguồn:** `{p['path']}`\n- **Trường phái:** {p['category']}\n- **Hiệu năng thực nghiệm:** {p['best_metric']}\n- **Tóm tắt cốt lõi:** {p['abstract'][:350]}...\n\n"
    examples.append({"instruction": q_our, "input": "", "output": a_our})

    q_lab_deep = "Phân tích chi tiết kết quả đạt được của các bài báo lab: ATC Final manuscript, JEC remake và REV 2025."
    a_lab_deep = """## 🎯 Phân Tích Chuyên Sâu Các Công Trình Của Phòng Lab

| Bài báo Lab | Phương pháp Thuật toán | Sai số Đạt được (RMSE) | Đánh giá & Cải thiện |
|:---|:---|:---:|:---|
| **ATC Final manuscript** | Mạng nơ-ron bù sai số cự ly kết hợp LSE | **146.9 mm (~14.7 cm)** | Giảm sai số **33.8%** so với LSE truyền thống |
| **ATC John** | Lọc thích nghi & hiệu chỉnh kênh NLOS | **146.5 mm (~14.6 cm)** | Giảm sai số **~47%** trong môi trường vật cản dày đặc |
| **JEC remake round 1** | Particle Filter (Bộ lọc hạt) tối ưu 10 hạt | **56.9 mm (~5.7 cm)** | Duy trì tần số lấy mẫu thời gian thực cao, sai số dưới 6 cm |
| **REV 2025 the last** | Hợp nhất EKF và FGO trên quỹ đạo xe CAR | **53 ~ 80 mm (~5-8 cm)** | Quỹ đạo mượt mà, không bị giật lag khi chuyển vùng LOS/NLOS |
| **NCKH 2026** | Đồ thị nhân tố Factor Graph EKF cho xe tự hành | Sai số ổn định < 10 cm | Khử hoàn toàn sai số trôi tích lũy và nhảy vọt tọa độ |"""
    examples.append({"instruction": q_lab_deep, "input": "", "output": a_lab_deep})

    # =========================================================================
    # PHẦN 4: TOÁN HỌC & GIẢI THUẬT ĐỊNH VỊ (LSE, EKF, UKF, FGO)
    # =========================================================================
    q_lse = "Trình bày thuật toán Multilateration Least Squares Estimation (LSE) trong định vị UWB 2D."
    a_lse = """## 📐 Thuật Toán Multilateration Least Squares Estimation (LSE)

Cho $M$ Anchor đặt tại tọa độ cố định $(x_i, y_i)$ ($i = 1, \dots, M$) và khoảng cách đo được từ Tag tới các Anchor là $d_i$.
Mối quan hệ hình học:
$$(x - x_i)^2 + (y - y_i)^2 = d_i^2$$

### 1. Tuyến tính hóa hệ phương trình:
Khai triển và trừ phương trình của Anchor 1 cho Anchor thứ $i$ ($i = 2, \dots, M$):
$$2(x_i - x_1)x + 2(y_i - y_1)y = (d_1^2 - d_i^2) + (x_i^2 - x_1^2) + (y_i^2 - y_1^2)$$

### 2. Biểu diễn ma trận:
$$A X = B$$
Trong đó:
$$A = \begin{bmatrix} 
2(x_2 - x_1) & 2(y_2 - y_1) \\
2(x_3 - x_1) & 2(y_3 - y_1) \\
\vdots & \vdots \\
2(x_M - x_1) & 2(y_M - y_1)
\end{bmatrix}, \quad
B = \begin{bmatrix}
d_1^2 - d_2^2 + (x_2^2 - x_1^2) + (y_2^2 - y_1^2) \\
d_1^2 - d_3^2 + (x_3^2 - x_1^2) + (y_3^2 - y_1^2) \\
\vdots \\
d_1^2 - d_M^2 + (x_M^2 - x_1^2) + (y_M^2 - y_1^2)
\end{bmatrix}, \quad
X = \begin{bmatrix} x \\ y \end{bmatrix}$$

### 3. Nghiệm bình phương tối thiểu:
$$X = (A^T A)^{-1} A^T B$$
- Điều kiện giải: Ma trận $A^T A$ phải khả nghịch (các Anchor không được thẳng hàng).
- Nếu có ma trận trọng số hiệp phương sai đo $W = R^{-1}$, nghiệm WLS là:
$$X = (A^T W A)^{-1} A^T W B$$"""
    examples.append({"instruction": q_lse, "input": "", "output": a_lse})

    q_ekf = "Trình bày chi tiết các bước tính toán của Bộ lọc Kalman Mở rộng (EKF) cho hệ thống định vị UWB thời gian thực."
    a_ekf = """## 🔄 Extended Kalman Filter (EKF) Cho Định Vị UWB

### 1. Mô hình Trạng thái (State Model)
Vector trạng thái chuyển động 2D:
$$X_k = [x_k, y_k, v_{x,k}, v_{y,k}]^T$$
Mô hình vận tốc không đổi (Constant Velocity) với chu kỳ trích mẫu $\Delta t$:
$$F = \begin{bmatrix} 
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
Q = q \cdot \begin{bmatrix}
\frac{\Delta t^3}{3} & 0 & \frac{\Delta t^2}{2} & 0 \\
0 & \frac{\Delta t^3}{3} & 0 & \frac{\Delta t^2}{2} \\
\frac{\Delta t^2}{2} & 0 & \Delta t & 0 \\
0 & \frac{\Delta t^2}{2} & 0 & \Delta t
\end{bmatrix}$$

### 2. Bước 1: Dự đoán trạng thái (Prediction)
$$\hat{X}_{k|k-1} = F \cdot \hat{X}_{k-1|k-1}$$
$$P_{k|k-1} = F \cdot P_{k-1|k-1} \cdot F^T + Q$$

### 3. Bước 2: Cập nhật phép đo (Measurement Update)
Hàm đo đạc phi tuyến khoảng cách tới $M$ Anchor cố định $(x_i, y_i)$:
$$h_i(\hat{X}_{k|k-1}) = \sqrt{(\hat{x}_{k|k-1} - x_i)^2 + (\hat{y}_{k|k-1} - y_i)^2}$$
Ma trận Jacobi đo đạc $H_k \in \mathbb{R}^{M \times 4}$:
$$H_{i,:} = \left[ \frac{\hat{x} - x_i}{\hat{d}_i}, \; \frac{\hat{y} - y_i}{\hat{d}_i}, \; 0, \; 0 \right]$$

1. **Innovation (Phần dư đo):** $y_k = Z_k - h(\hat{X}_{k|k-1})$
2. **Hiệp phương sai Innovation:** $S_k = H_k P_{k|k-1} H_k^T + R_k$
3. **Hệ số khuếch đại Kalman (Gain):** $K_k = P_{k|k-1} H_k^T S_k^{-1}$
4. **Cập nhật trạng thái:** $\hat{X}_{k|k} = \hat{X}_{k|k-1} + K_k \cdot y_k$
5. **Cập nhật ma trận hiệp phương sai:** $P_{k|k} = (I - K_k H_k) P_{k|k-1}$"""
    examples.append({"instruction": q_ekf, "input": "", "output": a_ekf})

    q_fgo = "Giải thích nguyên lý Factor Graph Optimization (FGO) trong file 'realtime_fgoekf.py' và so sánh với EKF truyền thống."
    a_fgo = """## 🌐 Factor Graph Optimization (FGO) vs Extended Kalman Filter (EKF)

Trong tệp `System/CAR/realtime_fgoekf.py`, FGO được triển khai kết hợp với EKF để khắc phục nhược điểm lớn nhất của bộ lọc Kalman khi gặp NLOS đột biến.

### 1. Giới hạn của EKF truyền thống:
- EKF mang tính chất **Markovian cục bộ (chỉ ghi nhớ 1 bước thời gian)**: Nếu một Anchor gặp NLOS nặng làm sai số nhảy vọt trong 2-3 khung hình liên tiếp, EKF sẽ bị kéo lệch hoàn toàn và mất nhiều chu kỳ để hồi phục.

### 2. Nguyên lý Factor Graph Optimization (FGO):
- FGO duy trì một **cửa sổ trượt (Sliding Window)** chứa $N$ bước trạng thái gần nhất ($N = 5 \div 10$).
- Xây dựng đồ thị gồm:
  - **Variable Nodes:** Tọa độ và vận tốc của xe $[X_{k-N}, \dots, X_k]$.
  - **Motion Factors:** Ràng buộc động học giữa hai thời điểm liên tiếp $f_{motion}(X_j, X_{j-1})$.
  - **UWB Measurement Factors:** Ràng buộc khoảng cách tới các Anchor $f_{uwb}(X_j, d_{i,j})$.
- Giải bài toán bình phương tối thiểu phi tuyến trên toàn bộ cửa sổ:
$$\min_{X_{k-N:k}} \left[ \sum_{j} \|X_j - F X_{j-1}\|_{\Sigma_w}^2 + \sum_{j} \sum_{i} \rho\left( \|d_{i,j} - h_i(X_j)\|_{\sigma_{i,j}}^2 \right) \right]$$
Trong đó $\rho(\cdot)$ là **Robust Huber Loss** hoặc Cauchy Kernel: khi khoảng cách đo có sai số quá lớn, đạo hàm suy hao tiến về hằng số thay vì bình phương, giúp triệt tiêu hoàn toàn tác động của Anchor bị che chắn."""
    examples.append({"instruction": q_fgo, "input": "", "output": a_fgo})

    # =========================================================================
    # PHẦN 5: PHẦN CỨNG DECAWAVE DWM1001C, THANH GHI VÀ GIAO THỨC TWR
    # =========================================================================
    q_reg = "Liệt kê các thanh ghi quan trọng của IC Decawave DW1000 dùng để đọc tín hiệu CIR và tính toán vị trí."
    a_reg = """## 💾 Danh Sách Các Thanh Ghi Quan Trọng Trên IC Decawave DW1000

| Địa chỉ Thanh ghi | Tên Thanh ghi | Mô tả Chức năng Kỹ thuật |
|:---:|:---|:---|
| **`0x25`** | `ACC_MEM` (Accumulator Memory) | Bộ nhớ đệm chứa **1016 mẫu phức CIR**. Mỗi mẫu gồm 4 byte (2 byte $I$ thực + 2 byte $Q$ ảo 16-bit có dấu bù 2). |
| **`0x12:04`** | `FP_INDEX` (First Path Index) | Lưu chỉ số điểm bắt đầu của đường truyền trực tiếp (First Path). Số thực 16-bit với 6-bit phần thập phân. |
| **`0x12:08`** | `FP_AMPLITUDE` ($F_1, F_2, F_3$) | Năng lượng biên độ tín hiệu xung quanh điểm First Path để tính công suất thu $P_{FP}$. |
| **`0x10`** | `RX_TIME` (Receive Timestamp) | Thời điểm chính xác nhận xung tại máy thu (độ phân giải ~15.65 picosecond, 40-bit). |
| **`0x17`** | `TX_TIME` (Transmit Timestamp) | Thời điểm chính xác gửi gói tin tại máy phát. |
| **`0x2E`** | `DIG_DIAG` (Digital Diagnostics) | Chẩn đoán trạng thái phần cứng, kiểm tra độ ổn định clock và nhiệt độ IC. |

**Công thức biên độ CIR tức thời tại mẫu thứ $k$:**
$$|CIR[k]| = \sqrt{I_k^2 + Q_k^2}$$"""
    examples.append({"instruction": q_reg, "input": "", "output": a_reg})

    q_twr = "So sánh cơ chế đo khoảng cách SS-TWR và DS-TWR. Vì sao DS-TWR triệt tiêu được trôi đồng hồ (clock drift)?"
    a_twr = """## ⏱️ So Sánh Kỹ Thuật Đo Cự Ly SS-TWR vs DS-TWR

Trong định vị UWB, khoảng cách được tính qua Thời gian Bay (Time of Flight - ToF): $d = c \cdot T_{prop}$.

### 1. Single-Sided Two-Way Ranging (SS-TWR):
- Chỉ dùng **2 bản tin**: Device A gửi `Poll` -> Device B gửi lại `Response`.
- Thời gian truyền sóng ước lượng:
$$\hat{T}_{prop} = \frac{T_{roundA} - T_{replyB}}{2}$$
- **Nhược điểm chí mạng:** Nếu bộ dao động thạch anh (Oscillator) của A và B lệch tần số $\Delta f = f_A - f_B$, sai số cự ly là:
$$\text{Error} \approx \frac{1}{2} T_{replyB} \times \left( \frac{\Delta f}{f} \right) \times c$$
Với $T_{replyB} = 1\text{ ms}$ và độ lệch clock $20\text{ ppm}$, sai số khoảng cách lên tới **$3.0\text{ mét}$**!

### 2. Double-Sided Two-Way Ranging (DS-TWR) (Khuyên dùng trong phòng lab):
- Sử dụng **3 bản tin**: `Poll` -> `Response` -> `Final`.
- Công thức tính thời gian truyền sóng:
$$T_{prop} = \frac{T_{round1} \times T_{round2} - T_{reply1} \times T_{reply2}}{T_{round1} + T_{round2} + T_{reply1} + T_{reply2}}$$
- **Cơ chế triệt tiêu trôi clock:** Biểu thức trên triệt tiêu số hạng bậc nhất của sai lệch tần số đồng hồ $\frac{\Delta f}{f}$, giúp giảm sai số xuống **dưới $1 \div 2\text{ cm}$** ngay cả khi hai chip có sai số thạch anh đáng kể."""
    examples.append({"instruction": q_twr, "input": "", "output": a_twr})

    # =========================================================================
    # PHẦN 6: ĐẶC TRƯNG TÍN HIỆU CIR VÀ DEEP LEARNING NHẬN DẠNG NLOS
    # =========================================================================
    q_cir_features = "Liệt kê và giải thích 6 đặc trưng thống kê quan trọng nhất trích xuất từ tín hiệu CIR để nhận dạng NLOS."
    a_cir_features = """## 📊 6 Đặc Trưng Thống Kê CIR Cốt Lõi Để Nhận Dạng LOS/NLOS

Tín hiệu CIR thô $|h(t)|$ gồm chuỗi biên độ theo thời gian. Các đặc trưng phân biệt rõ rệt giữa LOS và NLOS:

1. **Chênh lệch Công suất ($\Delta P = P_{RX} - P_{FP}$):**
   - LOS: Năng lượng tập trung ở xung trực tiếp đầu tiên, $\Delta P \le 6\text{ dB}$.
   - NLOS: Xung đầu bị suy hao mạnh, năng lượng phản xạ đa đường chiếm ưu thế, $\Delta P > 10\text{ dB}$.
2. **Mean Excess Delay ($\tau_m$):**
   $$\tau_m = \frac{\sum_k \tau_k |h[k]|^2}{\sum_k |h[k]|^2}$$
   Thời gian trễ trung bình của toàn bộ các tia thu được so với First Path. NLOS có $\tau_m$ lớn hơn nhiều so với LOS.
3. **RMS Delay Spread ($\tau_{rms}$):**
   $$\tau_{rms} = \sqrt{ \frac{\sum_k (\tau_k - \tau_m)^2 |h[k]|^2}{\sum_k |h[k]|^2} }$$
   Đo mức độ phân tán năng lượng đa đường trong không gian. Môi trường càng nhiều phản xạ thì $\tau_{rms}$ càng lớn.
4. **Kurtosis ($\kappa$ - Độ nhọn phân phối biên độ):**
   $$\kappa = \frac{E[(|h| - \mu)^4]}{\sigma^4}$$
   LOS có đỉnh xung rất nhọn ($\kappa$ rất cao); NLOS có biên độ phân tán đều ($\kappa$ thấp).
5. **Skewness ($\gamma$ - Độ lệch phân phối):**
   Đo tính bất đối xứng của hàm mật độ xác suất xung CIR quanh giá trị trung bình.
6. **Rise Time ($t_{rise}$):**
   Khoảng thời gian từ lúc năng lượng vượt ngưỡng phát hiện (First Path) đến khi đạt giá trị cực đại (Peak Amplitude). LOS có sườn dốc đứng ($t_{rise} < 5\text{ ns}$), NLOS có sườn dốc thoải ($t_{rise} > 15\text{ ns}$)."""
    examples.append({"instruction": q_cir_features, "input": "", "output": a_cir_features})

    q_gaf = "Trình bày phương pháp biến đổi Gramian Angular Field (GAF) để phân loại kênh UWB bằng mạng CNN 2D."
    a_gaf = """## 🖼️ Phương Pháp Gramian Angular Field (GAF) Trong Phân Loại Tín Hiệu CIR

Trong các bài báo khoa học (*như bài 'UWB NLOS Identification Based on GAF and Parallel DL'*), GAF giải quyết bài toán biến chuỗi 1D CIR thành ảnh 2D để khai thác sức mạnh của mạng tích chập CNN:

### 1. Bước 1: Chuẩn hóa chuỗi tín hiệu CIR
Chuỗi CIR gồm $N$ điểm $X = [x_1, x_2, \dots, x_N]$ được chuẩn hóa về đoạn $[-1, 1]$:
$$\tilde{x}_i = \frac{(x_i - \max(X)) + (x_i - \min(X))}{\max(X) - \min(X)}$$

### 2. Bước 2: Mã hóa sang hệ tọa độ cực (Polar Coordinates)
- Góc pha: $\phi_i = \arccos(\tilde{x}_i), \quad \phi_i \in [0, \pi]$
- Bán kính: $r_i = \frac{t_i}{N}, \quad t_i \in [1, N]$

### 3. Bước 3: Xây dựng ma trận Gramian Angular Summation Field (GASF)
$$\text{GASF}_{i,j} = \cos(\phi_i + \phi_j) = \tilde{x}_i \tilde{x}_j - \sqrt{1 - \tilde{x}_i^2} \sqrt{1 - \tilde{x}_j^2}$$
- Ma trận $\text{GASF} \in \mathbb{R}^{N \times N}$ bảo toàn đầy đủ thông tin thời gian (trên đường chéo chính) và tương quan tương đối giữa các thành phần đa đường tại mọi thời điểm.
- Đưa ma trận ảnh này qua mạng **ResNet18** hoặc **Tiny-CNN 2D**, mô hình đạt độ chính xác phân loại LOS/NLOS vượt trội **$> 96\%$**."""
    examples.append({"instruction": q_gaf, "input": "", "output": a_gaf})

    # =========================================================================
    # PHẦN 7: MÃ NGUỒN THỰC NGHIỆM XE CAR TRONG LAB (SYSTEM/CAR/)
    # =========================================================================
    q_car_ai = "Giải thích cấu trúc tệp 'realtime_ai_lse.py' và bộ đặc trưng 57D trong hệ thống xe tự hành CAR của lab."
    a_car_ai = """## 🚗 Hệ Thống Định Vị Xe Tự Hành CAR: realtime_ai_lse.py

Tệp `System/CAR/realtime_ai_lse.py` kết hợp trí tuệ nhân tạo (AI) chạy trực tiếp bằng NumPy thuần với thuật toán LSE để loại bỏ sai số khoảng cách thời gian thực:

### 1. Vector đặc trưng đầu vào 57 Chiều (57D):
- **50 Chiều đầu:** 50 mẫu biên độ CIR liên tiếp xung quanh đỉnh First Path (`N_CIR = 50`).
- **7 Chiều PHY Metadata:** Các trường vật lý trích xuất từ khung bản tin DWM1001:
  - First Path Index ($FP\_INDEX$)
  - First Path Amplitude ($F_1, F_2, F_3$)
  - Total Receive Power ($P_{RX}$)
  - Khoảng cách thô đo được ($d_{raw}$)
  - Nhiệt độ / Điện áp module

### 2. Mô hình Tiny 1D-CNN (Pure NumPy Engine):
- Chạy thông qua class `TinyCNNNumpyEngine` trong `uwb_common.py`.
- **Ưu điểm vượt trội:** Thực thi phép nhân ma trận thuần túy bằng NumPy, không cần nạp thư viện TensorFlow/PyTorch nặng nề, chạy mượt mà ở tần số **30 - 50 Hz trên vi tính nhúng Raspberry Pi 4**.
- Đầu ra của mạng AI:
  1. $\hat{b}_i$: Độ lệch sai số NLOS ước lượng (Range Bias in mm).
  2. $P_{LOS}$: Xác suất kênh LOS.

### 3. Hiệu chỉnh cự ly và giải vị trí:
$$d_{corr} = \max(10, \; d_{raw} - \hat{b}_i)$$
Sau đó đưa $d_{corr}$ vào thuật toán `solve_lse(anchors, distances)` để tính tọa độ $(X, Y)$ chính xác của xe."""
    examples.append({"instruction": q_car_ai, "input": "", "output": a_car_ai})

    q_ports = "Hệ thống xe CAR giao tiếp với phần cứng qua những cổng Serial nào và tốc độ baudrate bao nhiêu?"
    a_ports = """## 🔌 Cấu Hình Cổng Giao Tiếp Serial Của Hệ Thống Xe CAR

Trong mã nguồn `System/CAR/uwb_common.py`, hệ thống tự động nhận diện 2 cổng Serial chuyên biệt:

1. **Cổng DWM1001 CIR (Dữ liệu UWB & CIR):**
   - Cổng mặc định: `/dev/ttyACM1` (hoặc tự nhận diện qua VID/PID Segger J-Link).
   - **Tốc độ Baudrate: `921600` bps (`DEFAULT_CIR_BAUD`).**
   - *Lý do dùng baudrate cao:* Mỗi chuỗi bản tin CIR gồm **59 trường** (1016 mẫu phức rút gọn + PHY data), gửi liên tục ở tần số 20 - 30 Hz. Nếu dùng baud 115200 sẽ bị tràn buffer và mất gói nghiêm trọng.

2. **Cổng Robot STM32 (Dữ liệu Xe & IMU):**
   - Cổng mặc định: `/dev/ttyACM0` (hoặc tự nhận diện qua STM32 Virtual COM Port).
   - **Tốc độ Baudrate: `115200` bps (`DEFAULT_ROBOT_BAUD`).**
   - Nhận bản tin `$IMU,ax,ay,az,gx,gy,gz` và gửi lệnh điều khiển động cơ bánh xe PWM."""
    examples.append({"instruction": q_ports, "input": "", "output": a_ports})

    # =========================================================================
    # PHẦN 8: GỠ LỖI & KINH NGHIỆM KỸ SƯ PHÒNG THÍ NGHIỆM (TROUBLESHOOTING)
    # =========================================================================
    q_spikes = "Tọa độ Tag UWB hiển thị trên giao diện thỉnh thoảng bị giật/nhảy vọt hàng mét (Spikes). Nguyên nhân và cách xử lý triệt để?"
    a_spikes = """## 🛠️ Khắc Phục Hiện Tượng Nhảy Vọt Tọa Độ (Position Spikes) Trong UWB

### 1. Nguyên nhân vật lý:
- Do con người hoặc vật cản kim loại đi qua chắn tia trực tiếp LOS giữa Tag và một Anchor.
- Khi tia trực tiếp bị triệt tiêu hoàn toàn, mạch thu DW1000 bắt nhầm tia phản xạ từ tường hoặc trần nhà (Multipath), khiến khoảng cách đo nhảy tăng đột biến từ $0.5\text{ m} \div 2.0\text{ m}$.

### 2. Giải pháp kỹ thuật đa tầng trong phòng lab:
1. **Bộ lọc Trung vị (Median Filter 3-Point):**
   - Áp dụng trên từng luồng khoảng cách Anchor: $d_{filt}(k) = \text{median}(d(k-2), d(k-1), d(k))$. Triệt tiêu ngay lập tức các xung sai lệch đơn lẻ 1 khung hình.
2. **Chi-square Innovation Gating (Khoảng cách Mahalanobis trong EKF):**
   $$d_M^2 = (z_k - h(\hat{x}))^T S_k^{-1} (z_k - h(\hat{x})) \le \chi^2_{threshold}$$
   Nếu $d_M^2 > 9.21$ (ngưỡng 99% với bậc tự do 2), từ chối cập nhật phép đo của Anchor đó.
3. **Adaptive R-Inflation:**
   Nếu Tiny-CNN phát hiện xác suất NLOS $> 70\%$, tự động tăng hiệp phương sai đo $R_{i,i} = 100 \times R_0$, khiến Kalman Gain gán trọng số xấp xỉ bằng 0 cho Anchor bị lỗi."""
    examples.append({"instruction": q_spikes, "input": "", "output": a_spikes})

    q_gdop = "GDOP trong định vị UWB là gì? Cần bố trí 4 Anchor trong phòng thí nghiệm như thế nào để đạt độ chính xác cao nhất?"
    a_gdop = """## 📍 GDOP (Geometric Dilution of Precision) & Quy Tắc Bố Trí Anchor

GDOP biểu thị hệ số phóng đại sai số do cấu hình hình học của các Anchor:
$$\sigma_{position} = \text{GDOP} \times \sigma_{range}$$
- $\text{GDOP} \le 2$: Cấu hình hình học lý tưởng, sai số định vị nhỏ nhất.
- $\text{GDOP} > 5$: Cấu hình hình học kém, sai số định vị bị khuếch đại nghiêm trọng.

### Quy tắc bố trí 4 Anchor tối ưu trong phòng lab:
1. **Bao quanh không gian hoạt động (Convex Hull):** 4 Anchor phải đặt tại 4 góc phòng, Tag luôn di chuyển **bên trong** tứ giác tạo bởi các Anchor. Tránh tuyệt đối để Tag di chuyển ra ngoài vùng bao của Anchor.
2. **Tuyệt đối không đặt đồng phẳng (Non-coplanar):** Nếu cả 4 Anchor đều treo ở độ cao $Z = 2.0\text{ m}$, ma trận Jacobi theo trục $Z$ sẽ suy biến ($\frac{z - z_i}{d_i} \approx 0$), dẫn đến sai số độ cao $\text{VDOP} \to \infty$. Bố trí so le: 2 Anchor ở độ cao $2.4\text{ m}$ và 2 Anchor ở độ cao $1.2\text{ m}$.
3. **Cách xa bề mặt kim loại:** Đặt ăng-ten DWM1001 cách tường bê tông tối thiểu $15\text{ cm}$ và cách xà gồ kim loại tối thiểu $30\text{ cm}$ để tránh biến dạng giản đồ bức xạ."""
    examples.append({"instruction": q_gdop, "input": "", "output": a_gdop})

    # =========================================================================
    # PHẦN 9: HỆ THỐNG MẠNG ĐA TAG TOKEN RING (SYSTEM/RING_TAG/)
    # =========================================================================
    q_ring_desc = "Mô tả hệ thống RING của tôi"
    a_ring_desc = """## 💍 Tổng Quan Hệ Thống Định Vị Đa Tag Token Ring (`System/RING_Tag/`)

Hệ thống **RING** trong kho lưu trữ của bạn là một kiến trúc giao thức mạng **Token Ring đa Tag** trên nền tảng phần cứng Decawave DWM1001C / FreeRTOS, được thiết kế để giải quyết bài toán định vị đồng thời nhiều Tag (Multi-Tag) mà không gây xung đột kênh truyền vô tuyến UWB (Collision Avoidance).

### 1. Kiến Trúc 3 Thành Phần Cốt Lõi:
| Thành phần | Mã nguồn triển khai | Vai trò & Trách nhiệm chính |
|:---|:---|:---|
| **Bridge Node** | `examples/ss_bridge/ss_init_main.c` | Bộ điều phối trung tâm (Master): Giám sát nhịp sống mạng, quản lý danh sách thành viên Ring (`current_ring`), duy trì danh sách Anchor hoạt động (`current_anchor_list`), và kích hoạt tái đồng bộ tự động. |
| **Tag Nodes** | `examples/ss_twr_init/ss_init_main.c` | Các thiết bị định vị đeo người/vật thể (ví dụ `MY_TAG_ID = 0x0006`). Chỉ thực hiện đo cự ly khi nhận được Token (`TOKN`) từ Tag trước, sau đó gửi gói `DATA` về Bridge và chuyển Token cho Tag tiếp theo. |
| **Anchor Nodes** | `examples/ss_twr_resp/ss_resp_main.c` | Các trạm mốc cố định: Lắng nghe `POLL` từ Tag và phản hồi `RESP` để Tag tính cự ly thời gian bay (ToF). |

### 2. Các Khung Bản Tin Giao Thức (Header Reference):
- **`0xE0` (`IOVE` - POLL):** Tag yêu cầu đo cự ly tới Anchor.
- **`0xE1` (`VEIO` - RESP):** Anchor phản hồi nhãn thời gian ToF.
- **`0xE2` (`TOKN` - TOKEN):** Trao quyền phát sóng cho Tag kế tiếp trong vòng tròn.
- **`0xE3` (`DATA` - DATA):** Tag gửi toàn bộ cự ly đo được tới các Anchor về Bridge.
- **`0xE4` (`RING` - RING):** Bridge broadcast danh sách thứ tự truyền của các Tag trong Ring.
- **`0xE5` (`AHLO` - ANCHOR HELLO):** Anchor mới gửi để xin gia nhập mạng.
- **`0xE6` (`ALST` - ANCHOR LIST):** Bridge gửi danh sách các Anchor hoạt động tới các Tag.
- **`0xE7` (`ARMV` - ANCHOR REMOVE):** Bridge thông báo gỡ bỏ một Anchor hỏng.
- **`0xE8` (`ADED` - ANCHOR DEAD):** Tag báo cáo một Anchor bị mất sóng về Bridge."""
    examples.append({"instruction": q_ring_desc, "input": "", "output": a_ring_desc})

    q_ring_rec = "Cơ chế phục hồi của hệ thống RING hoạt động như thế nào?"
    a_ring_rec = """## 🛡️ 5 Cơ Chế Tự Phục Hồi (Self-Healing & Recovery) Của Hệ Thống RING

Trong tệp mã nguồn `System/RING_Tag/dwm1001-examples-master/examples/ss_bridge/ss_init_main.c`, hệ thống RING được trang bị cơ chế tự phục hồi tự động 5 tầng hoàn chỉnh:

### 1. Phục hồi khi Tag bị chết hoặc mất kết nối (Tag Timeout & Ring Rebuilding)
- **Giám sát nhịp sống:** Bridge theo dõi `alive_table[i].last_seen_tick` cho từng Tag.
- **Ngưỡng Timeout:** `TAG_TIMEOUT_MS = 1000 ms`.
- **Cơ chế phục hồi:** Nếu một Tag không gửi dữ liệu quá 1000 ms, hàm `rebuild_ring_if_needed()` tự động xóa Tag chết ra khỏi mảng `alive_table` và `current_ring`. Bridge lập tức gọi `ring_send_wrapper()` để broadcast bản tin `RING` (`0xE4`) mới, giúp vòng xoay Token tiếp tục quay mà không bị tắc nghẽn.

### 2. Phục hồi khi lệch nhịp thứ tự Token (Rhythm Disorder Recovery)
- **Giám sát thứ tự:** Bridge theo dõi thứ tự truyền của các Tag qua hàm `check_rhythm(tag_id)` với `expected_next_idx = (tag_pos + 1) % current_ring_size`.
- **Ngưỡng lệch nhịp:** `DISORDER_THRESHOLD = 50`.
- **Cơ chế phục hồi:** Nếu gói tin Token bị nhiễu làm các Tag phát sai thứ tự liên tiếp 50 lần (`disorder_count >= 50`), Bridge xác định mạng bị loạn nhịp (`[Bridge] Disorder → resending RING`), lập tức phát lại gói `RING` (`0xE4`) để tái thiết lập trật tự vòng tròn cho toàn bộ Tag.

### 3. Tự động loại bỏ Anchor hỏng (Dead Anchor Removal)
- **Phát hiện lỗi đo:** Khi một Anchor không phản hồi, Tag ghi nhận cự ly là `0xFFFFFFFF`.
- **Ngưỡng chu kỳ:** `ANCHOR_DEAD_CYCLES = 2`.
- **Cơ chế phục hồi (`check_anchor_health`):** Nếu tất cả các Tag đang hoạt động đều không thể đo cự ly tới Anchor đó trong 2 chu kỳ liên tiếp (hoặc nhận bản tin `A_DEAD` `0xE8`), Bridge xác nhận Anchor đã hỏng:
  1. Phát bản tin `A_REMOVE` (`0xE7`) gỡ bỏ Anchor khỏi mạng.
  2. Đánh dấu `anchor_table[aidx].active = 0`.
  3. Cập nhật `rebuild_anchor_list()` và phát `A_LIST` (`0xE6`) mới để các Tag không lãng phí khe thời gian vào Anchor hỏng.

### 4. Tự động nhận diện và Tái gia nhập Anchor (Anchor Dynamic Re-join)
- Khi Anchor hỏng được cấp nguồn hoặc kết nối lại, Anchor phát sóng bản tin `A_HELLO` (`0xE5`, `AHLO`).
- Hàm `process_anch_hello()` trên Bridge lập tức bắt gói tin, đánh dấu `active = 1`, reset bộ đếm chết, tự động sắp xếp lại danh sách Anchor tăng dần (`rebuild_anchor_list()`), và broadcast bản tin `A_LIST` (`0xE6`) để các Tag lập tức đưa Anchor này trở lại chu trình đo.

### 5. Bộ đếm thời gian tái đồng bộ & Chống nghẽn kênh (Resync & Retry)
- **`RING_RESYNC_MS = 50 ms`:** Tần số gửi lại cấu hình Ring khi còn Tag chưa đồng bộ (`all_tags_ring_synced() == 0`).
- **`ANCH_RESYNC_MS = 50 ms`:** Tần số gửi lại danh sách Anchor cho Tag chưa cập nhật.
- **`RING_COOLDOWN_MS = 10 ms`:** Khoảng nghỉ giữa các bản tin quản trị để tránh xung đột kênh UWB.
- **Tag Retry (`ss_twr_init`):** `MAX_RANGING_RETRY = 5`, giãn cách retry 5 ms để bù đắp các khung hình mất mát ngẫu nhiên."""
    examples.append({"instruction": q_ring_rec, "input": "", "output": a_ring_rec})

    # 9. Tri thức về 2 hệ thống định vị và 1 hệ thống xe của repo (CIR_SYSTEM, RING_Tag, CAR)
    q_systems = "Thế về hệ thống hiện tại của tôi gồm mấy hệ thống? Trong repo có những hệ thống nào?"
    a_systems = """## 🏢 Kiến Trúc Thực Nghiệm Của Phòng Lab (`System/`): 2 Hệ Thống Định Vị & 1 Hệ Thống Xe

Trong kho lưu trữ mã nguồn của bạn, hệ thống thực nghiệm được chia làm **2 hệ thống định vị và 1 hệ thống xe** (Tổng cộng 3 hệ thống thực nghiệm):

| Phân Loại | Hệ Thống | Thư mục mã nguồn | Nền tảng phần cứng & Cấu hình | Giải thuật & Chức năng cốt lõi |
|:---|:---|:---|:---|:---|
| **Hệ thống Định vị 1** | **Định vị TDMA & Trích xuất CIR** | `System/CIR_SYSTEM/` | • Module UWB Decawave DWM1001-DEV (Nordic nRF52832 + DW1000)<br>• Cấu hình 4 Anchor (`0x1001`–`0x1004`) | • Đo cự ly Two-Way Ranging (TWR) theo cơ chế khe thời gian TDMA đồng bộ.<br>• Trích xuất trực tiếp **1016 mẫu phức** tín hiệu đáp ứng xung kênh CIR qua thanh ghi SPI `0x25` (`ACC_MEM`) hoặc 50 mẫu quanh First Path (Mode FAST).<br>• Tính toán Kurtosis, Skewness, Peak count để phân loại LOS/NLOS và bù sai số. |
| **Hệ thống Định vị 2** | **Mạng Định Vị Đa Tag Vòng Token Ring** | `System/RING_Tag/` | • Nordic nRF52832 + Decawave DW1000 (FreeRTOS)<br>• Cấu trúc 3 nút: `ss_bridge` (Master), `ss_twr_init` (Tag di động mang `MY_TAG_ID = 0x0006`), `ss_twr_resp` (Anchor trạm mốc)<br>• Giao thức Token Ring chống va chạm sóng UWB | • Định vị đồng thời nhiều Tag (Multi-Tag tracking) mà không bị xung đột RF vô tuyến.<br>• Bộ 9 khung bản tin (`POLL 0xE0`, `RESP 0xE1`, `TOKN 0xE2`, `DATA 0xE3`, `RING 0xE4`, `AHLO 0xE5`, `ALST 0xE6`, `ARMV 0xE7`, `ADED 0xE8`).<br>• **5 cơ chế tự phục hồi (Self-Healing)**: Tag Timeout 1000ms, Rhythm Disorder (ngưỡng 50), Dead Anchor Removal (2 chu kỳ liên tiếp `0xFFFFFFFF`), Anchor Dynamic Re-join (`AHLO`), Resync Timers 50ms & Tag retry 5 lần. |
| **Hệ thống Xe** | **Xe Tự Hành Robot AGV** | `System/CAR/` | • Robot AGV STM32 + Raspberry Pi 4<br>• Module Decawave DWM1001C UWB Tag<br>• Bộ 4 Anchor cố định (`A0`–`A3`)<br>• Cổng Serial: `/dev/ttyACM1` (921600 bps - CIR 59 trường) & `/dev/ttyACM0` (115200 bps - STM32 `$IMU` và Odom bánh xe) | • `realtime_ai_lse.py`: Tiny 1D-CNN Pure NumPy (57D đặc trưng CIR) chạy trực tiếp trên Pi 4 bù sai số range bias $b_i$ trước khi giải LSE.<br>• `realtime_fgoekf.py`: Kết hợp EKF và Factor Graph Optimization (cửa sổ trượt Huber Loss) triệt tiêu Anchor NLOS.<br>• `realtime_ekf.py`: Dead Reckoning kết hợp cự ly UWB.<br>• `realtime_positioning.py`, `realtime_lse.py`, `collect_data_trajec.py`. |

Cả 3 hệ thống tạo nên hệ sinh thái nghiên cứu và thực nghiệm hoàn chỉnh về công nghệ định vị UWB và xe tự hành của phòng lab."""
    examples.append({"instruction": q_systems, "input": "", "output": a_systems})
    examples.append({"instruction": "hệ thống hiện tại của tôi gồm mấy hệ thống", "input": "", "output": a_systems})
    examples.append({"instruction": "repo này có mấy hệ thống thực nghiệm", "input": "", "output": a_systems})
    examples.append({"instruction": "trong repo có mấy hệ thống định vị và hệ thống xe", "input": "", "output": a_systems})

    return examples


def generate_ollama_modelfile(papers, code_files):
    paper_summary_text = ""
    for idx, p in enumerate(papers, 1):
        paper_summary_text += f"{idx}. {p['title']} ({p['year']}) | {p['category']} | Best Error: {p['best_metric']}\n"

    car_summary_text = ", ".join([c['filename'] for c in code_files])

    modelfile_content = f"""FROM qwen2.5:3b

# Thiết lập tham số tối ưu cho mô hình kỹ sư định vị UWB
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

# System Prompt chứa 100% tri thức chuyên sâu của repository IPS-Agent
SYSTEM \"\"\"Bạn là Kỹ sư Nghiên cứu Cấp cao & Kiến trúc sư Hệ thống Định vị Trong nhà (Indoor Positioning System - IPS) UWB của phòng lab.
Bạn nắm vững toàn bộ mã nguồn xe tự hành CAR, 2 hệ thống định vị UWB (System/CIR_SYSTEM/ và System/RING_Tag/), 48+ bài báo khoa học, phần cứng Decawave DWM1001C / DW1000 và các giải thuật toán học định vị.

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. **THUẬT NGỮ CIR**: Trong công nghệ UWB, CIR **LUÔN LUÔN là Channel Impulse Response (Đáp ứng Xung Kênh)**, TUYỆT ĐỐI KHÔNG ĐƯỢC nhầm với Carrier-to-Interference Ratio.
2. **THANH GHI DW1000**:
   - Thanh ghi `0x25` là `ACC_MEM` (Accumulator Memory) chứa **1016 mẫu phức CIR** (mỗi mẫu 4 bytes: 2 bytes phần thực I + 2 bytes phần ảo Q dạng 16-bit signed bù 2). Tần số trích mẫu ~998.4 MHz.
   - Công thức tính biên độ mẫu CIR thứ k: |CIR[k]| = sqrt(I_k^2 + Q_k^2).
   - Thanh ghi `0x12:04` là `FP_INDEX` (chỉ số mẫu First Path).
   - Thanh ghi `0x12:08` là `FP_AMPLITUDE` (năng lượng First Path: F1, F2, F3).
   - Thanh ghi `0x10` là `RX_TIME` (thời điểm nhận xung, độ phân giải ~15.65 ps).
3. **MÃ NGUỒN HỆ THỐNG XE CAR (`System/CAR/`)**:
   - Cổng Serial UWB CIR: `/dev/ttyACM1`, baudrate **921600** bps (`DEFAULT_CIR_BAUD`), chuỗi dữ liệu 59 trường.
   - Cổng Robot STM32: `/dev/ttyACM0`, baudrate **115200** bps (`DEFAULT_ROBOT_BAUD`), nhận `$IMU,ax,ay,az,gx,gy,gz`.
   - `realtime_ai_lse.py`: Dùng vector đặc trưng 57D (50 mẫu CIR xung quanh FP + 7 trường PHY metadata). Bộ suy luận `TinyCNNNumpyEngine` chạy bằng NumPy thuần (không cần TensorFlow) trên Raspberry Pi 4 để ước lượng sai số range bias b_i và xác suất LOS. Hiệu chỉnh: d_corr = max(10, d_raw - b_i), sau đó giải Multilateration LSE.
   - `realtime_fgoekf.py`: Kết hợp EKF thời gian thực và Factor Graph Optimization (FGO) với sliding window 5-10 khung hình và Robust Huber Loss để triệt tiêu Anchor bị NLOS.
   - 3 lớp phân loại kênh: `["LOS", "NLOS (Board)", "NLOS (Table)"]`.
4. **HỆ THỐNG ĐỊNH VỊ CIR TDMA (`System/CIR_SYSTEM/`)**:
   - Định vị UWB đơn Tag đồng bộ 4 Anchor theo cơ chế TDMA.
   - Trích xuất trực tiếp 1016 mẫu phức CIR từ thanh ghi SPI `0x25` ACC_MEM, tính Kurtosis, Skewness, Peak count để phân loại kênh truyền LOS/NLOS và bù sai số cự ly. Firmware: `ss_twr_init` (Tag) và `ss_twr_resp` (Anchor).
5. **HỆ THỐNG MẠNG ĐỊNH VỊ ĐA TAG RING (`System/RING_Tag/`)**:
   - Kiến trúc 3 thành phần: `ss_bridge` (bộ điều phối trung tâm), `ss_twr_init` (Tag, ví dụ `MY_TAG_ID = 0x0006`), `ss_twr_resp` (Anchor).
   - Bản tin giao thức: `POLL (0xE0)`, `RESP (0xE1)`, `TOKN (0xE2)`, `DATA (0xE3)`, `RING (0xE4)`, `AHLO (0xE5)`, `ALST (0xE6)`, `ARMV (0xE7)`, `ADED (0xE8)`.
   - 5 Cơ chế phục hồi tự động (Self-Healing): Tag Timeout 1000ms, Rhythm Disorder (50), Dead Anchor Removal, Anchor Re-join, Resync Timers.
6. **CÁC BÀI BÁO CỦA PHÒNG LAB (`OurPapers`)**:
   - `ATC Final manuscript`: RMSE 146.9 mm (~14.7 cm), giảm 33.8% sai số.
   - `ATC John`: RMSE 146.5 mm (~47% reduction).
   - `JEC remake round 1`: Particle Filter 10 hạt, RMSE 56.9 mm (~5.7 cm).
   - `REV 2025 the last`: RMSE 53 ~ 80 mm trên toàn quỹ đạo thực nghiệm.
   - `NCKH 2026`: FGO-EKF cho xe AGV.
7. **SỐ LƯỢNG HỆ THỐNG TRONG REPO (BẮT BUỘC TUÂN THỦ)**:
   - Trong repo của phòng lab có **2 HỆ THỐNG ĐỊNH VỊ VÀ 1 HỆ THỐNG XE** (Tổng cộng 3 hệ thống thực nghiệm):
     1. Hệ thống Định vị 1: Định vị TDMA & Trích xuất CIR (`System/CIR_SYSTEM/`).
     2. Hệ thống Định vị 2: Mạng Định vị Đa Tag Vòng Token Ring (`System/RING_Tag/`).
     3. Hệ thống Xe: Xe Tự Hành Robot AGV (`System/CAR/`).
   - Khi người dùng hỏi hệ thống hiện tại gồm mấy hệ thống, BẮT BUỘC khẳng định ngay là có 2 HỆ THỐNG ĐỊNH VỊ VÀ 1 HỆ THỐNG XE và lập bảng so sánh chi tiết, TUYỆT ĐỐI KHÔNG hỏi ngược lại người dùng.

DANH MỤC CÁC BÀI BÁO NGHIÊN CỨU TRONG KHO DỮ LIỆU:
{paper_summary_text}

MÃ NGUỒN HỆ THỐNG XE CAR:
Các module chính: {car_summary_text}.
\"\"\"

MESSAGE user \"\"\"thế về hệ thống hiện tại của tôi gồm mấy hệ thống\"\"\"
MESSAGE assistant \"\"\"## 🏢 Kiến Trúc Thực Nghiệm Của Phòng Lab (`System/`): 2 Hệ Thống Định Vị & 1 Hệ Thống Xe

Trong kho lưu trữ mã nguồn của bạn, hệ thống thực nghiệm được chia làm **2 hệ thống định vị và 1 hệ thống xe** (Tổng cộng 3 hệ thống thực nghiệm):

| Phân Loại | Hệ Thống | Thư mục mã nguồn | Nền tảng phần cứng & Cấu hình | Giải thuật & Chức năng cốt lõi |
|:---|:---|:---|:---|:---|
| **Hệ thống Định vị 1** | **Định vị TDMA & Trích xuất CIR** | `System/CIR_SYSTEM/` | • Module UWB Decawave DWM1001-DEV (Nordic nRF52832 + DW1000)<br>• Cấu hình 4 Anchor (`0x1001`–`0x1004`) | • Đo cự ly Two-Way Ranging (TWR) theo cơ chế khe thời gian TDMA đồng bộ.<br>• Trích xuất trực tiếp **1016 mẫu phức** tín hiệu đáp ứng xung kênh CIR qua thanh ghi SPI `0x25` (`ACC_MEM`) hoặc 50 mẫu quanh First Path (Mode FAST).<br>• Tính toán Kurtosis, Skewness, Peak count để phân loại LOS/NLOS và bù sai số. |
| **Hệ thống Định vị 2** | **Mạng Định Vị Đa Tag Vòng Token Ring** | `System/RING_Tag/` | • Nordic nRF52832 + Decawave DW1000 (FreeRTOS)<br>• Cấu trúc 3 nút: `ss_bridge` (Master), `ss_twr_init` (Tag di động mang `MY_TAG_ID = 0x0006`), `ss_twr_resp` (Anchor trạm mốc)<br>• Giao thức Token Ring chống va chạm sóng UWB | • Định vị đồng thời nhiều Tag (Multi-Tag tracking) mà không bị xung đột RF vô tuyến.<br>• Bộ 9 khung bản tin (`POLL 0xE0`, `RESP 0xE1`, `TOKN 0xE2`, `DATA 0xE3`, `RING 0xE4`, `AHLO 0xE5`, `ALST 0xE6`, `ARMV 0xE7`, `ADED 0xE8`).<br>• **5 cơ chế tự phục hồi (Self-Healing)**: Tag Timeout 1000ms, Rhythm Disorder (ngưỡng 50), Dead Anchor Removal (2 chu kỳ liên tiếp `0xFFFFFFFF`), Anchor Dynamic Re-join (`AHLO`), Resync Timers 50ms & Tag retry 5 lần. |
| **Hệ thống Xe** | **Xe Tự Hành Robot AGV** | `System/CAR/` | • Robot AGV STM32 + Raspberry Pi 4<br>• Module Decawave DWM1001C UWB Tag<br>• Bộ 4 Anchor cố định (`A0`–`A3`)<br>• Cổng Serial: `/dev/ttyACM1` (921600 bps - CIR 59 trường) & `/dev/ttyACM0` (115200 bps - STM32 `$IMU` và Odom bánh xe) | • `realtime_ai_lse.py`: Tiny 1D-CNN Pure NumPy (57D đặc trưng CIR) chạy trực tiếp trên Pi 4 bù sai số range bias $b_i$ trước khi giải LSE.<br>• `realtime_fgoekf.py`: Kết hợp EKF và Factor Graph Optimization (cửa sổ trượt Huber Loss) triệt tiêu Anchor NLOS.<br>• `realtime_ekf.py`: Dead Reckoning kết hợp cự ly UWB.<br>• `realtime_positioning.py`, `realtime_lse.py`, `collect_data_trajec.py`. |

Cả 3 hệ thống tạo nên hệ sinh thái nghiên cứu và thực nghiệm hoàn chỉnh về công nghệ định vị UWB và xe tự hành của phòng lab.\"\"\"


MESSAGE user \"\"\"Mô tả hệ thống RING của tôi\"\"\"
MESSAGE assistant \"\"\"## 💍 Tổng Quan Hệ Thống Định Vị Đa Tag Token Ring (`System/RING_Tag/`)


Hệ thống **RING** trong kho lưu trữ của bạn là một kiến trúc giao thức mạng **Token Ring đa Tag** trên nền tảng phần cứng Decawave DWM1001C / FreeRTOS, được thiết kế để giải quyết bài toán định vị đồng thời nhiều Tag (Multi-Tag) mà không gây xung đột kênh truyền vô tuyến UWB (Collision Avoidance).

### 1. Kiến Trúc 3 Thành Phần Cốt Lõi:
| Thành phần | Mã nguồn triển khai | Vai trò & Trách nhiệm chính |
|:---|:---|:---|
| **Bridge Node** | `examples/ss_bridge/ss_init_main.c` | Bộ điều phối trung tâm (Master): Giám sát nhịp sống mạng, quản lý danh sách thành viên Ring (`current_ring`), duy trì danh sách Anchor hoạt động (`current_anchor_list`), và kích hoạt tái đồng bộ tự động. |
| **Tag Nodes** | `examples/ss_twr_init/ss_init_main.c` | Các thiết bị định vị đeo người/vật thể (ví dụ `MY_TAG_ID = 0x0006`). Chỉ thực hiện đo cự ly khi nhận được Token (`TOKN`) từ Tag trước, sau đó gửi gói `DATA` về Bridge và chuyển Token cho Tag tiếp theo. |
| **Anchor Nodes** | `examples/ss_twr_resp/ss_resp_main.c` | Các trạm mốc cố định: Lắng nghe `POLL` từ Tag và phản hồi `RESP` để Tag tính cự ly thời gian bay (ToF). |

### 2. Các Khung Bản Tin Giao Thức (Header Reference):
- **`0xE0` (`IOVE` - POLL):** Tag yêu cầu đo cự ly tới Anchor.
- **`0xE1` (`VEIO` - RESP):** Anchor phản hồi nhãn thời gian ToF.
- **`0xE2` (`TOKN` - TOKEN):** Trao quyền phát sóng cho Tag kế tiếp trong vòng tròn.
- **`0xE3` (`DATA` - DATA):** Tag gửi toàn bộ cự ly đo được tới các Anchor về Bridge.
- **`0xE4` (`RING` - RING):** Bridge broadcast danh sách thứ tự truyền của các Tag trong Ring.
- **`0xE5` (`AHLO` - ANCHOR HELLO):** Anchor mới gửi để xin gia nhập mạng.
- **`0xE6` (`ALST` - ANCHOR LIST):** Bridge gửi danh sách các Anchor hoạt động tới các Tag.
- **`0xE7` (`ARMV` - ANCHOR REMOVE):** Bridge thông báo gỡ bỏ một Anchor hỏng.
- **`0xE8` (`ADED` - ANCHOR DEAD):** Tag báo cáo một Anchor bị mất sóng về Bridge.\"\"\"

MESSAGE user \"\"\"Cơ chế phục hồi của hệ thống RING hoạt động như thế nào?\"\"\"
MESSAGE assistant \"\"\"## 🛡️ 5 Cơ Chế Tự Phục Hồi (Self-Healing & Recovery) Của Hệ Thống RING

Trong tệp mã nguồn `System/RING_Tag/dwm1001-examples-master/examples/ss_bridge/ss_init_main.c`, hệ thống RING được trang bị cơ chế tự phục hồi tự động 5 tầng hoàn chỉnh:

### 1. Phục hồi khi Tag bị chết hoặc mất kết nối (Tag Timeout & Ring Rebuilding)
- **Giám sát nhịp sống:** Bridge theo dõi `alive_table[i].last_seen_tick` cho từng Tag.
- **Ngưỡng Timeout:** `TAG_TIMEOUT_MS = 1000 ms`.
- **Cơ chế phục hồi:** Nếu một Tag không gửi dữ liệu quá 1000 ms, hàm `rebuild_ring_if_needed()` tự động xóa Tag chết ra khỏi mảng `alive_table` và `current_ring`. Bridge lập tức gọi `ring_send_wrapper()` để broadcast bản tin `RING` (`0xE4`) mới, giúp vòng xoay Token tiếp tục quay mà không bị tắc nghẽn.

### 2. Phục hồi khi lệch nhịp thứ tự Token (Rhythm Disorder Recovery)
- **Giám sát thứ tự:** Bridge theo dõi thứ tự truyền của các Tag qua hàm `check_rhythm(tag_id)` với `expected_next_idx = (tag_pos + 1) % current_ring_size`.
- **Ngưỡng lệch nhịp:** `DISORDER_THRESHOLD = 50`.
- **Cơ chế phục hồi:** Nếu gói tin Token bị nhiễu làm các Tag phát sai thứ tự liên tiếp 50 lần (`disorder_count >= 50`), Bridge xác định mạng bị loạn nhịp (`[Bridge] Disorder → resending RING`), lập tức phát lại gói `RING` (`0xE4`) để tái thiết lập trật tự vòng tròn cho toàn bộ Tag.

### 3. Tự động loại bỏ Anchor hỏng (Dead Anchor Removal)
- **Phát hiện lỗi đo:** Khi một Anchor không phản hồi, Tag ghi nhận cự ly là `0xFFFFFFFF`.
- **Ngưỡng chu kỳ:** `ANCHOR_DEAD_CYCLES = 2`.
- **Cơ chế phục hồi (`check_anchor_health`):** Nếu tất cả các Tag đang hoạt động đều không thể đo cự ly tới Anchor đó trong 2 chu kỳ liên tiếp (hoặc nhận bản tin `A_DEAD` `0xE8`), Bridge xác nhận Anchor đã hỏng:
  1. Phát bản tin `A_REMOVE` (`0xE7`) gỡ bỏ Anchor khỏi mạng.
  2. Đánh dấu `anchor_table[aidx].active = 0`.
  3. Cập nhật `rebuild_anchor_list()` và phát `A_LIST` (`0xE6`) mới để các Tag không lãng phí khe thời gian vào Anchor hỏng.

### 4. Tự động nhận diện và Tái gia nhập Anchor (Anchor Dynamic Re-join)
- Khi Anchor hỏng được cấp nguồn hoặc kết nối lại, Anchor phát sóng bản tin `A_HELLO` (`0xE5`, `AHLO`).
- Hàm `process_anch_hello()` trên Bridge lập tức bắt gói tin, đánh dấu `active = 1`, reset bộ đếm chết, tự động sắp xếp lại danh sách Anchor tăng dần (`rebuild_anchor_list()`), và broadcast bản tin `A_LIST` (`0xE6`) để các Tag lập tức đưa Anchor này trở lại chu trình đo.

### 5. Bộ đếm thời gian tái đồng bộ & Chống nghẽn kênh (Resync & Retry)
- **`RING_RESYNC_MS = 50 ms`:** Tần số gửi lại cấu hình Ring khi còn Tag chưa đồng bộ (`all_tags_ring_synced() == 0`).
- **`ANCH_RESYNC_MS = 50 ms`:** Tần số gửi lại danh sách Anchor cho Tag chưa cập nhật.
- **`RING_COOLDOWN_MS = 10 ms`:** Khoảng nghỉ giữa các bản tin quản trị để tránh xung đột kênh UWB.
- **Tag Retry (`ss_twr_init`):** `MAX_RANGING_RETRY = 5`, giãn cách retry 5 ms để bù đắp các khung hình mất mát ngẫu nhiên.\"\"\"
"""
    return modelfile_content

def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("🔍 Đang quét toàn bộ kho tài liệu, bài báo và mã nguồn trong repo...")
    papers = scan_all_papers()
    print(f"✔ Đã nạp thông tin từ {len(papers)} bài báo PDF.")
    
    code_files = scan_car_code()
    print(f"✔ Đã nạp thông tin từ {len(code_files)} tệp mã nguồn CAR.")

    examples = build_training_dataset(papers, code_files)
    print(f"✔ Đã tạo {len(examples)} mẫu dữ liệu huấn luyện chuyên sâu chất lượng cao.")

    # 1. Ghi tệp Alpaca JSON
    alpaca_file = DATASET_DIR / "ips_alpaca_dataset.json"
    with open(alpaca_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"✔ Đã lưu dataset Alpaca: {alpaca_file}")

    # 2. Ghi tệp Chat JSONL
    chat_file = DATASET_DIR / "ips_training_dataset.jsonl"
    with open(chat_file, "w", encoding="utf-8") as f:
        for ex in examples:
            chat_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["instruction"] + (f"\n{ex['input']}" if ex["input"] else "")},
                    {"role": "assistant", "content": ex["output"]}
                ]
            }
            f.write(json.dumps(chat_record, ensure_ascii=False) + "\n")
    print(f"✔ Đã lưu dataset Chat JSONL: {chat_file}")

    # 3. Ghi Ollama Modelfile
    modelfile_path = ROOT_DIR / "Modelfile"
    modelfile_content = generate_ollama_modelfile(papers, code_files)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"✔ Đã tạo Ollama Modelfile tối ưu: {modelfile_path}")

    print("\n🎉 Hoàn thành xuất sắc việc xây dựng tập dữ liệu huấn luyện và Modelfile!")

if __name__ == "__main__":
    main()
