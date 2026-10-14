# 📡 IPS-UWB AI Agent Terminal

Trợ lý Kỹ thuật & Kỹ sư Nghiên cứu Ảo chuyên sâu về **Hệ thống Định vị Trong nhà (Indoor Positioning System - UWB)**, tích hợp sẵn trong kho lưu trữ mã nguồn.

Giao diện tương tác trực tiếp ngay trong **Terminal** (phong cách AI Coding Assistant / Antigravity), hiển thị Markdown đẹp mắt, công thức toán học và trích dẫn tài liệu chính xác.

---

## 🌟 Tính năng nổi bật

1. **Kho tri thức UWB nạp sẵn (3.600+ đoạn trích chuyên sâu)**:
   - **Tài liệu phần cứng DWM1001C** (`Document/`): Datasheet, Firmware API Guide, User Guide, PANS firmware, cấu hình TLV UART/SPI.
   - **54 Bài báo khoa học quốc tế** (`Paper/`): Phân loại LOS/NLOS, trích xuất đặc trưng CIR (Channel Impulse Response), EKF/UKF/FGO-EKF, Fusion UWB + IMU/LiDAR.
   - **Mã nguồn thực nghiệm của dự án** (`System/`): Thuật toán định vị CAR (`realtime_ekf.py`, `realtime_fgoekf.py`, `realtime_ai_lse.py`...), CIR System và Tag.
2. **Hệ thống Kỹ năng Định hình Mô hình (`skills/` - Chuẩn Antigravity)**:
   - Toàn bộ hành vi, tư duy kỹ sư cấp cao, quy tắc kẻ bảng `box.ROUNDED`, và chuyên môn được đóng gói vào các tệp Markdown trong `skills/`.
   - Bất kỳ ai cũng có thể chỉnh sửa tệp `SKILL.md` để dạy model cách suy luận, định dạng và trích dẫn theo ý muốn.
   - Gõ lệnh `/skills` trong terminal để xem danh sách kỹ năng đang nạp.
3. **Linh hoạt 2 chế độ (Không lo bị Rate Limit)**:
   - **Chế độ LOCAL (Ollama)**: Chạy 100% trên máy tính cá nhân, **không giới hạn câu hỏi**, **không cần internet**, **không tốn phí**.
   - **Chế độ ONLINE (Groq / Gemini)**: Phản hồi cực nhanh, tư duy chuyên sâu, hỗ trợ giải thích bài báo học thuật phức tạp.
4. **Bộ công cụ (Tools) tích hợp**:
   - Tự động tra cứu và trích dẫn chính xác bài báo khoa học liên quan.
   - Quét 100% toàn bộ kho dữ liệu bài báo không bỏ sót bài nào (`/all <chủ đề>`).
   - Công cụ giải toán hình học vị trí Tag 2D bằng Least Squares Trilateration (`/calc`).
   - Tìm kiếm nhanh tức thì bằng từ khóa (`/search <từ khóa>`).


---

## 🚀 Hướng dẫn khởi chạy (Dành cho bất kỳ ai clone repo về)

### 1. Cài đặt thư viện cần thiết (Chỉ mất 5 giây)
```bash
pip install -r requirements_agent.txt
```

### 2. Khởi chạy Agent

* **Trên Linux / macOS**:
  ```bash
  ./run_agent.sh
  # Hoặc: python3 ips_agent.py
  ```

* **Trên Windows**:
  ```cmd
  run_agent.bat
  :: Hoặc: python ips_agent.py
  ```

---

## ⚙️ Hướng dẫn chọn chế độ

### Chế độ 1: Dùng LOCAL (Không giới hạn, 0đ - Khuyên dùng lâu dài)
Nếu bạn đã cài [Ollama](https://ollama.com/), hãy tải một mô hình nhẹ:
```bash
ollama pull qwen2.5:3b     # Khuyên dùng: ~2GB, tư duy thuật toán và tiếng Việt cực tốt
# Hoặc: ollama pull llama3.2:3b
```
Sau đó chỉ cần khởi động `./run_agent.sh`, Agent sẽ tự động kết nối và chat không giới hạn!

### Chế độ 2: Dùng ONLINE (Google Gemini)
Nếu không muốn tải model về máy, bạn có thể dùng API:
1. Lấy API key miễn phí tại: [Google AI Studio](https://aistudio.google.com/)
2. Chạy với tham số:
   ```bash
   ./run_agent.sh --mode online --api-key YOUR_API_KEY
   ```
   *(Hoặc xuất biến môi trường: `export GEMINI_API_KEY=YOUR_API_KEY`)*

---

## ⌨️ Các lệnh điều khiển trong Terminal

| Lệnh | Chức năng |
| :--- | :--- |
| `/help` | Xem danh sách tất cả các lệnh điều khiển |
| `/key <api_key>` | Thêm API Key tạm thời cho phiên chat (tự động xóa khi tắt terminal, gõ `/key clear` để gỡ) |
| `/mode` | Chuyển đổi qua lại giữa chế độ Server, Local (Ollama) và Online |
| `/model <tên>` | Đổi mô hình LLM (VD: `/model qwen2.5:7b` hoặc `/model gemini-2.5-flash`) |
| `/search <từ khóa>` | Tra cứu trực tiếp trong kho 54 bài báo và datasheet không qua LLM |
| `/calc` | Tính toán tọa độ Tag 2D bằng thuật toán Least Squares |
| `/index` | Quét và lập chỉ mục lại toàn bộ tài liệu và mã nguồn mới |
| `/clear` | Xóa sạch màn hình terminal |
| `/exit` hoặc `/quit` | Thoát khỏi Agent |

---

## 📂 Cấu trúc thư mục

```
IPS/
├── Document/                 # Datasheet, API Guide DWM1001
├── Paper/                    # 54 bài báo khoa học UWB/CIR/NLOS/EKF
├── System/                   # Mã nguồn thực tế hệ thống CAR, CIR, Tag
├── ips_agent.py              # Điểm khởi chạy chính của Agent
├── run_agent.sh              # Script 1-click cho Linux/macOS
├── run_agent.bat             # Script 1-click cho Windows
├── requirements_agent.txt    # Thư viện tối giản (rich, requests, numpy)
└── agent_core/
    ├── agent.py              # Logic điều phối và Prompt chuyên gia UWB
    ├── kb_engine.py          # Bộ máy lập chỉ mục & tìm kiếm tri thức siêu tốc
    ├── tools.py              # Công cụ tính toán và tra cứu
    ├── llm_provider.py       # Kết nối Ollama và Gemini API
    └── ui.py                 # Giao diện Terminal mang phong cách Antigravity
```
