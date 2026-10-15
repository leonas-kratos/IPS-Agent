---
name: dwm1001-hardware-expert
description: Kỹ năng chuyên sâu về phần cứng Decawave DWM1001C, giao thức SPI/UART TLV, thanh ghi đọc CIR và mã nguồn xe CAR/Tag trong dự án.
version: 1.0.0
---

# Decawave DWM1001C Hardware & Firmware Integration Skill

## 1. Thông số Phần cứng Cốt lõi của Module DWM1001C
* **IC Thu phát UWB:** Decawave DW1000 (Tuân thủ chuẩn IEEE 802.15.4-2011 UWB).
* **Vi điều khiển chính:** Nordic Semiconductor nRF52832 (ARM Cortex-M4 32-bit, 64 MHz, BLE tích hợp).
* **Cảm biến chuyển động:** STMicroelectronics LIS2DH12 (Gia tốc kế 3 trục tiêu thụ cực thấp).
* **Băng tần UWB:** Kênh tần số Channel 5 (Tần số trung tâm 6489.6 MHz, Băng thông 499.2 MHz).
* **Độ chính xác đo:** $\pm 10\text{ cm}$ trong điều kiện LOS tiêu chuẩn.
* **Tốc độ truyền dữ liệu UWB:** Mặc định 6.8 Mbps (hỗ trợ 110 kbps và 850 kbps).

---

## 2. Kỹ thuật Đọc Tín hiệu CIR từ Thanh ghi DW1000
Để huấn luyện các mô hình AI/Deep Learning nhận dạng NLOS, cần đọc thô dữ liệu CIR qua giao diện SPI:
1. **Thanh ghi Bộ nhớ Đệm CIR:** Địa chỉ thanh ghi `0x25` (`ACC_MEM` - Accumulator Memory).
2. **Kích thước Dữ liệu:** Gồm **1016 mẫu phức** (Complex Samples) tại tần số lấy mẫu $\sim 998.4\text{ MHz}$ (khoảng cách giữa các mẫu xấp xỉ $1.0016\text{ ns}$).
3. **Định dạng Mỗi Mẫu (4 bytes):**
   * Byte 0-1: Phần thực $I$ (16-bit Signed Integer, bù 2).
   * Byte 2-3: Phần ảo $Q$ (16-bit Signed Integer, bù 2).
4. **Tính toán Biên độ Tín hiệu (Amplitude Envelope):**
   $$|CIR[k]| = \sqrt{I_k^2 + Q_k^2}$$

---

## 3. Kiến trúc 2 Hệ Thống Triển Khai Thực Tế Cốt Lõi (Thư mục `System/`)
Dự án IPS UWB của phòng lab bao gồm **2 hệ thống thực nghiệm cốt lõi**:
* **1. Hệ thống Xe Tự Hành CAR (`System/CAR/`):**
  * Xe AGV trang bị vi điều khiển STM32 kết nối máy tính nhúng Raspberry Pi 4 và module UWB Tag Decawave DWM1001C, kết nối bộ 4 Anchor (`A0`–`A3`).
  * Cổng Serial: `/dev/ttyACM1` (921600 baud, 59 trường CIR) và `/dev/ttyACM0` (115200 baud, STM32 `$IMU` và Odom bánh xe).
  * Các thuật toán định vị thời gian thực: `realtime_ai_lse.py` (Tiny 1D-CNN Pure NumPy 57D bù sai số range bias trước khi giải LSE), `realtime_fgoekf.py` (EKF + Sliding Window FGO với Huber Loss triệt tiêu NLOS), `realtime_ekf.py`, `realtime_positioning.py`.
* **2. Hệ thống Mạng Đa Tag Vòng Token Ring (`System/RING_Tag/`):**
  * Giải quyết bài toán định vị đồng thời nhiều Tag (Multi-Tag tracking) không va chạm sóng vô tuyến UWB.
  * Cấu trúc 3 loại Node trên nRF52832 + DW1000 FreeRTOS: `ss_bridge` (Master điều phối), `ss_twr_init` (Tag di động mang `MY_TAG_ID = 0x0006`, chỉ đo TWR khi cầm Token `0xE2`), `ss_twr_resp` (Anchor trạm mốc).
  * 5 cơ chế tự phục hồi (Self-Healing): Tag Timeout 1000ms, Rhythm Disorder (ngưỡng 50), Dead Anchor Removal (2 chu kỳ 0xFFFFFFFF), Anchor Dynamic Re-join (`AHLO`), Resync Timers 50ms & Tag retry 5 lần.
* **3. Phân hệ bổ trợ CIR_SYSTEM (`System/CIR_SYSTEM/`):**
  * Trích xuất liên tục 1016 mẫu phức CIR qua thanh ghi SPI `0x25` ACC_MEM của DW1000 để huấn luyện các mạng Deep Learning nhận dạng NLOS.

