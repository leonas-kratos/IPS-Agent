# 🚀 Hướng dẫn Đưa Máy Chủ AI lên Hugging Face (Dùng SDK Gradio)

Hugging Face Spaces hỗ trợ sẵn **Gradio** (lựa chọn mặc định và phổ biến nhất, **không cần Docker**, **không cần thẻ ngân hàng**, miễn phí 100%).

Server này vừa cung cấp **Giao diện Web Chat trực quan**, vừa cung cấp **API luôn online** cho Terminal CLI (`./run_agent.sh`).

---

### Bước 1: Tạo một Space mới trên Hugging Face
1. Vào [huggingface.co/new-space](https://huggingface.co/new-space).
2. Điền các trường:
   * **Space name**: `ips-uwb-server` (hoặc tên tùy thích).
   * **License**: `mit`.
   * **Select the Space SDK**: Chọn **Gradio** (Tùy chọn mặc định đầu tiên!).
   * **Space hardware**: Giữ nguyên **CPU basic • 2 vCPU • 16GB RAM • FREE** (Miễn phí 100%).
3. Bấm nút **Create Space**.

---

### Bước 2: Tải 2 file lên Space
Sau khi tạo Space xong, bạn vào tab **Files** -> Bấm **Add file** -> **Upload files**, kéo thả 2 file sau:
1. `server/app.py`
2. `server/requirements.txt`

Bấm **Commit changes to main**.

---

### Bước 3: Thêm API Key bí mật
Tại trang Space vừa tạo:
1. Vào tab **Settings** -> Cuộn xuống tìm mục **Variables and secrets**.
2. Bấm nút **New secret**:
   * **Name**: `GROQ_API_KEY` (hoặc `GEMINI_API_KEY`)
   * **Value**: Dán API key của bạn vào:
     * *Khuyên dùng Groq vì cực nhanh và miễn phí: lấy tại [console.groq.com/keys](https://console.groq.com/keys).*
     * *Hoặc lấy Gemini key tại [aistudio.google.com](https://aistudio.google.com).*
3. Bấm **Save**.

Sau 1 phút, Space sẽ chuyển sang màu xanh lá **Running**.

---

### Bước 4: Lấy URL gắn vào Repo của bạn
1. Trên trang Space, nhìn lên góc trên bên phải, bấm vào nút **3 chấm (...)** -> Chọn **Embed this Space**.
2. Copy đường link trong mục **Direct URL** (link có dạng: `https://<ten-user>-ips-uwb-server.hf.space`).
3. Mở file [agent_core/config.py](file:///home/leonas/Documents/AllData/IPS/agent_core/config.py) trong repo của bạn và dán URL vào:
   ```python
   DEFAULT_SERVER_URL = "https://<ten-user>-ips-uwb-server.hf.space"
   ```

---

🎉 **Hoàn tất!**
* Người muốn chat trên web: Chỉ cần vào link Space đó là chat được ngay.
* Người clone repo về: Gõ `./run_agent.sh` trong terminal là kết nối tự động tới server để chat!
