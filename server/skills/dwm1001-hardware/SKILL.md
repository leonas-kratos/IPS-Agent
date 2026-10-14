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

## 3. Kiến trúc Hệ thống Triển khai Thực tế (Thư mục `System/`)
Dự án IPS UWB của phòng lab bao gồm các hệ thống phần cứng và phần mềm sau:
* **Hệ thống Xe Tự hành CAR (`System/CAR_SYSTEM/`):**
  * Xe trang bị Tag UWB giao tiếp với bộ 4 Anchor cố định đặt tại 4 góc phòng.
  * Tích hợp encoder bánh xe, IMU và vi điều khiển điều khiển động cơ qua PWM.
* **Hệ thống CIR_SYSTEM (`System/CIR_SYSTEM/`):**
  * Module đọc liên tục chuỗi xung CIR tốc độ cao và truyền về máy tính/server.
  * Tích hợp script định vị `realtime_ekf.py` và `realtime_fgoekf.py`.
* **Hệ thống Vòng đeo RING Tag (`System/RING_Tag/`):**
  * Thiết bị đeo tay/đeo người kích thước nhỏ gọn phục vụ bài toán giám sát vị trí bệnh nhân hoặc công nhân trong nhà máy.
