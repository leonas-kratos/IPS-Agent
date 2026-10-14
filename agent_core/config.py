import os

# Cấu hình máy chủ tập trung (Always Online Server) trên Hugging Face
DEFAULT_SERVER_URL = os.environ.get(
    "IPS_SERVER_URL",
    "https://leonaskratos-ips-uwb-server.hf.space"
)

# Groq API Key dự phòng khi cần dùng Cloud (Lấy từ biến môi trường GROQ_API_KEY)
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Chế độ mặc định:
# 'local': Chạy 100% trên phần cứng máy tính (Ollama) - 0% GIỚI HẠN, 0đ, KHÔNG CẦN MẠNG
# 'online': Chạy qua Groq Cloud siêu tốc
# 'server': Chạy qua Hugging Face Space
DEFAULT_MODE = "local"

# Tên mô hình mặc định (Custom Model thuần đã nướng 100% tri thức của repo)
DEFAULT_MODEL = "ips-uwb"

# Chế độ RAG mặc định (False = Chạy thuần Pure LLM, không trích xuất đoạn chunk vụn vặt)
DEFAULT_USE_RAG = False
