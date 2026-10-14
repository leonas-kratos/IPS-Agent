---
name: sensor-fusion-ekf
description: Kỹ năng chuyên sâu về Bộ lọc Kalman Mở rộng (EKF), Unscented Kalman (UKF), Factor Graph Optimization (FGO) và hợp nhất cảm biến UWB + IMU.
version: 1.0.0
---

# Sensor Fusion & State Estimation Skill (EKF / UKF / FGO)

## 1. Mô hình Toán học EKF trong Hệ thống IPS UWB
Trong định vị UWB thời gian thực, đo khoảng cách $d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2}$ là hàm phi tuyến của tọa độ Tag. Bộ lọc Kalman tuyến tính chuẩn không áp dụng được trực tiếp, cần dùng **Extended Kalman Filter (EKF)**.

### Vector Trạng thái (State Vector)
$$X_k = [p_x, p_y, p_z, v_x, v_y, v_z]^T$$
* $p_x, p_y, p_z$: Tọa độ 3D của Tag tại thời điểm $k$.
* $v_x, v_y, v_z$: Vận tốc 3D của Tag tại thời điểm $k$.

### Bước 1: Dự đoán Trạng thái (Prediction Step)
Mô hình chuyển động vận tốc không đổi (Constant Velocity Model):
$$X_{k|k-1} = F \cdot X_{k-1|k-1} + B \cdot u_k$$
$$P_{k|k-1} = F \cdot P_{k-1|k-1} \cdot F^T + Q$$
Trong đó:
* $F = \begin{bmatrix} I_3 & \Delta t \cdot I_3 \\ 0_3 & I_3 \end{bmatrix}$ là ma trận chuyển trạng thái.
* $Q$: Ma trận hiệp phương sai nhiễu quá trình (Process Noise Covariance).

### Bước 2: Cập nhật Đo đạc (Measurement Update Step)
Hàm đo đạc phi tuyến cho $M$ Anchor cố định tại $(x_i, y_i, z_i)$:
$$h_i(X_k) = \sqrt{(p_x - x_i)^2 + (p_y - y_i)^2 + (p_z - z_i)^2}$$
Ma trận đạo hàm riêng Jacobi $H_k$:
$$H_k = \begin{bmatrix} \frac{\partial h_1}{\partial p_x} & \frac{\partial h_1}{\partial p_y} & \frac{\partial h_1}{\partial p_z} & 0 & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ \frac{\partial h_M}{\partial p_x} & \frac{\partial h_M}{\partial p_y} & \frac{\partial h_M}{\partial p_z} & 0 & 0 & 0 \end{bmatrix}$$
Với:
$$\frac{\partial h_i}{\partial p_x} = \frac{p_x - x_i}{\hat{d}_i}, \quad \frac{\partial h_i}{\partial p_y} = \frac{p_y - y_i}{\hat{d}_i}, \quad \frac{\partial h_i}{\partial p_z} = \frac{p_z - z_i}{\hat{d}_i}$$

### Bước 3: Tính toán Kalman Gain và Cập nhật
1. Độ lệch đo (Innovation): $y_k = Z_k - h(X_{k|k-1})$
2. Hiệp phương sai Innovation: $S_k = H_k P_{k|k-1} H_k^T + R_k$
3. Hệ số khuếch đại Kalman (Gain): $K_k = P_{k|k-1} H_k^T S_k^{-1}$
4. Cập nhật trạng thái: $X_{k|k} = X_{k|k-1} + K_k \cdot y_k$
5. Cập nhật hiệp phương sai: $P_{k|k} = (I - K_k H_k) P_{k|k-1}$

---

## 2. Kỹ thuật Thích nghi khi gặp NLOS (Adaptive EKF)
Khi một Anchor bị phát hiện NLOS hoặc có sai số thô (Outlier):
* **Chiến lược R-Inflation:** Tăng giá trị phương sai đo $R_{i,i} = \alpha \cdot R_{0}$ (với $\alpha = 100 \div 1000$) để Kalman Gain tương ứng $K_i \to 0$. Khi đó phép đo của Anchor này hầu như không ảnh hưởng đến ước lượng vị trí.
* **Chiến lược Chi-square Gating:** Kiểm tra khoảng cách Mahalanobis $d_M^2 = y_k^T S_k^{-1} y_k$. Nếu $d_M^2 > \chi^2_{threshold}$, loại bỏ ngay phép đo.

---

## 3. Mã nguồn thực tế trong Repository
Trong thư mục `System/` của dự án đã có sẵn các cài đặt EKF hoàn chỉnh:
* `System/CIR_SYSTEM/realtime_ekf.py`: Triển khai EKF thời gian thực nhận dữ liệu qua Serial/UDP.
* `System/CIR_SYSTEM/realtime_fgoekf.py`: Kết hợp EKF trực tuyến với Factor Graph Optimization chạy ngầm để làm mượt quỹ đạo xe CAR.
