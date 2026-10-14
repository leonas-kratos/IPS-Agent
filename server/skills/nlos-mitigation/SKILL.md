---
name: nlos-mitigation-expert
description: Kỹ năng chuyên sâu về nhận dạng kênh truyền LOS/NLOS và các thuật toán giảm thiểu sai số NLOS (NLOS Error Mitigation) trên tín hiệu UWB CIR.
version: 1.0.0
---

# NLOS Identification & Mitigation Expert Skill

## 1. Cơ chế Vật lý của Hiện tượng NLOS trong UWB
Trong truyền sóng vô tuyến siêu băng rộng (UWB):
* **LOS (Line-of-Sight):** Đường truyền thẳng trực tiếp giữa Tag và Anchor không bị che chắn. Xung First Path (FP) xuất hiện đầu tiên và có năng lượng cao nhất.
* **NLOS (Non-Line-of-Sight):** Đường truyền thẳng bị chắn bởi tường, kính, người hoặc vật thể kim loại:
  1. *Xuyên qua vật cản:* Vận tốc truyền sóng giảm ($v = c/\sqrt{\epsilon_r}$ với $\epsilon_r > 1$), tạo ra độ trễ thời gian $\Delta \tau > 0$ và sai số dương khoảng cách $d_{meas} = d_{true} + b_{NLOS}$ ($b_{NLOS} > 0$).
  2. *Phản xạ đa đường:* Tia đầu tiên bị chặn hoàn toàn, máy thu bắt nhầm tia phản xạ gián tiếp làm tăng đột biến khoảng cách đo.

---

## 2. Các Đặc trưng Cốt lõi của Tín hiệu CIR (Channel Impulse Response)
Khi phân tích dữ liệu DWM1001 hoặc bộ dữ liệu CIR 1016 mẫu:
1. **First Path Index:** Vị trí mẫu đầu tiên mà năng lượng vượt ngưỡng phát hiện.
2. **First Path Power ($F_1, F_2, F_3$):** Năng lượng tại 3 mẫu đầu xung quanh FP index.
3. **Total Receive Power ($P_{RX}$):** Tổng công suất thu toàn dải CIR.
4. **Tỷ số Công suất:** $\Delta P = P_{RX} - P_{FP}$. Trong môi trường LOS, $\Delta P \approx 0$ đến $6\text{ dB}$. Trong môi trường NLOS, $\Delta P > 10\text{ dB}$.
5. **Rise Time:** Thời gian từ lúc bắt đầu có năng lượng đến khi đạt đỉnh Max Peak. LOS có sườn dốc đứng; NLOS có sườn dốc thoải.
6. **Kurtosis & Skewness:** Độ nhọn và độ lệch của phân phối biên độ xung CIR.

---

## 3. Bản đồ các Trường phái Kỹ thuật Mitigation trong Kho dữ liệu
Khi được hỏi về bài toán Mitigation, bạn phải đối chiếu theo 4 hướng tiếp cận chính:

### Trường phái 1: Deep Learning & Mạng Nơ-ron (End-to-End Mitigation)
* **Kỹ thuật Gramian Angular Field (GAF):** Chuyển chuỗi xung CIR 1D thành ma trận ảnh 2D giữ nguyên quan hệ pha và thời gian. Đưa vào CNN phân loại kênh với độ chính xác $> 96\%$ (*Bài báo 'UWB NLOS Mitigation Based on GAF and Parallel DL'*).
* **Parallel Deep Learning (CNN + LSTM):** Một nhánh CNN phân loại kênh, một nhánh LSTM hồi quy trực tiếp giá trị sai số $\hat{b}_{NLOS}$ để trừ ra khỏi khoảng cách đo (*Bài báo 'Parallel Deep Learning for NLOS Detection and Error Mitigation'*).
* **Meta-Learning (MAML) & Transfer Learning:** Giúp mô hình thích ứng nhanh với môi trường mới chỉ với 5-10 mẫu (Few-shot learning), giải quyết vấn đề Hard Samples (*Bài báo 'Hard Sample Meta-Learning for CIR NLOS Identification'*).

### Trường phái 2: Bộ lọc Thích nghi & Tối ưu hóa (Adaptive Filtering & FGO)
* **Adaptive EKF / Robust Kalman:** Điều chỉnh động ma trận hiệp phương sai sai số đo $R_k$. Khi phát hiện NLOS, tăng mạnh hiệp phương sai $R_k \to \infty$ để giảm trọng số của Anchor đó trong bước cập nhật đo (*Bài báo 'A Tightly Coupled Adaptive EKF for Indoor 3-D Positioning'*).
* **VB-CSRUKF & RTS Smoother:** Kết hợp lọc Kalman biến phân với bộ làm mượt hai chiều RTS, cho sai số cực thấp trong hầm lò ngầm (*Bài báo 'Improving Localization Accuracy Using VB-CSRUKF and RTS'*).
* **Factor Graph Optimization (FGO):** Xây dựng đồ thị nhân tố và áp dụng hàm mất mát Cauchy / Huber Robust Kernel để triệt tiêu ảnh hưởng của các Anchor dị biệt (Outliers) (*Bài báo nhóm Lab: NCKH_2026.pdf*).

### Trường phái 3: Hợp nhất Đa cảm biến (Sensor Fusion)
* **UWB + PDR (Pedestrian Dead Reckoning):** Dùng gia tốc kế và con quay hồi chuyển đếm bước chân và ước lượng hướng đi để bù đắp khi toàn bộ UWB rơi vào vùng che khuất hoàn toàn (*Bài báo 'Enhanced Indoor Pedestrian Tracking Using UWB/PDR Fusion'*).
* **UWB + MIMU:** Đạt RMSE ấn tượng chỉ **4.0 cm** (*Bài báo 'Compensating the NLoS Occlusion Errors of UWB With MIMU'*).

### Trường phái 4: Bản đồ Kênh & Hình học (Anchor Geometry & RIS)
* **Bản đồ Anchor LOS/NLOS Map:** Khảo sát trước vùng che khuất trong tòa nhà để hiệu chỉnh động ma trận hình học (*Bài báo 'An Indoor UWB NLOS Correction Positioning Method Based on Anchor Map'*).
* **Dynamic Anchor Selection:** Đánh giá tính khả thi hình học (GDOP/HDOP) và loại bỏ các Anchor NLOS làm xấu hình học định vị (*Bài báo 'Dynamic Real-Time Anchor Selection'*).
