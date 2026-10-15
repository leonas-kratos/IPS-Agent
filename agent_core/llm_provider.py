import os
import json
import requests
from typing import Generator, List, Dict, Any, Optional
from agent_core.config import DEFAULT_SERVER_URL, DEFAULT_GROQ_API_KEY, DEFAULT_MODEL

class LLMProvider:
    def __init__(self, mode: str = "online", model: str = None, api_key: str = None, server_url: str = None):
        self.mode = mode.lower()  # 'online', 'server', hoặc 'local'
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or DEFAULT_GROQ_API_KEY
        self.server_url = server_url or os.environ.get("IPS_SERVER_URL", DEFAULT_SERVER_URL)
        self.ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def get_ollama_models(self) -> List[str]:
        """Lấy danh sách các model đang có sẵn trong Ollama."""
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if res.status_code == 200:
                data = res.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def check_connection(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái kết nối tới backend."""
        if self.mode == "server":
            try:
                res = requests.get(f"{self.server_url}/health", timeout=3)
                if res.status_code == 200:
                    return {
                        "ok": True,
                        "type": "Always-Online Central Server (Hugging Face)",
                        "server_url": self.server_url
                    }
            except Exception:
                pass
            return {
                "ok": False,
                "type": "Always-Online Server (Chưa kết nối)",
                "server_url": self.server_url,
                "error": "Server chưa sẵn sàng."
            }

        elif self.mode == "local":
            try:
                res = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                if res.status_code == 200:
                    models = [m["name"] for m in res.json().get("models", [])]
                    return {
                        "ok": True,
                        "type": "Ollama Local (0 Limit, 0$)",
                        "models": models,
                        "current_model": self.model or (models[0] if models else "Chưa có model (cần ollama pull)")
                    }
            except Exception:
                pass
            return {
                "ok": False,
                "type": "Ollama Local",
                "error": f"Không kết nối được tới {self.ollama_url}."
            }

        else: # direct online
            if not self.api_key:
                return {
                    "ok": False,
                    "type": "Online API",
                    "error": "Chưa cấu hình API Key (gõ /key để thêm)."
                }
            
            provider = "Groq (Llama-3.3-70B)" if self.api_key.startswith("gsk_") else ("Google Gemini" if self.api_key.startswith("AIzaSy") else "OpenAI Compatible")
            return {
                "ok": True,
                "type": f"Online API: {provider}",
                "current_model": self.model or ("llama-3.3-70b-versatile" if self.api_key.startswith("gsk_") else "gemini-2.5-flash")
            }

    def chat_stream(self, system_prompt: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Gửi tin nhắn và stream câu trả lời về terminal."""
        if self.mode == "server":
            yield from self._stream_server(system_prompt, messages)
        elif self.mode == "local":
            yield from self._stream_ollama(system_prompt, messages)
        else:
            yield from self._stream_online(system_prompt, messages)

    def _stream_free_unlimited(self, system_prompt: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Kênh AI dự phòng miễn phí."""
        url = "https://text.pollinations.ai/openai/chat/completions"
        all_msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": "openai",
            "messages": all_msgs,
            "stream": True,
            "temperature": 0.3
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    yield f"\n[Lỗi kết nối AI: {r.status_code}]\n"
                    return
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    content = choices[0]["delta"].get("content", "")
                                    if content:
                                        yield content
                            except Exception:
                                continue
        except Exception as e:
            yield f"\n[Lỗi kết nối mạng: {str(e)}]\n"

    def _stream_server(self, system_prompt: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        url = f"{self.server_url}/api/chat"
        all_msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "messages": all_msgs,
            "stream": True
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=8) as r:
                if r.status_code != 200:
                    yield f"\n[💡 Máy chủ ({r.status_code}) chưa sẵn sàng -> Đang tự động kết nối Kênh AI Dự phòng...]\n\n"
                    yield from self._stream_free_unlimited(system_prompt, messages)
                    return
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            try:
                                chunk = json.loads(data_str)
                                content = chunk.get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue
        except Exception:
            yield f"\n[💡 Không thể kết nối máy chủ -> Đang tự động kết nối Kênh AI Dự phòng...]\n\n"
            yield from self._stream_free_unlimited(system_prompt, messages)

    def _stream_ollama(self, system_prompt: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        url = f"{self.ollama_url}/api/chat"
        models = self.get_ollama_models()
        selected_model = self.model or ("ips-uwb" if "ips-uwb:latest" in models or "ips-uwb" in models else ("qwen2.5:3b" if "qwen2.5:3b" in models else (models[0] if models else None)))

        if not selected_model or not models:
            yield "\n[⚠️ Ollama chưa tải model 'qwen2.5:3b'. Tự động chuyển sang Kênh AI Dự phòng (Để dùng offline, hãy chạy: 'ollama pull qwen2.5:3b')...]\n\n"
            yield from self._stream_free_unlimited(system_prompt, messages)
            return

        # Master Skill cốt lõi: Đảm bảo mô hình nắm vững 2 hệ thống cốt lõi và các quy tắc kỹ thuật
        effective_sys = """Bạn là Kỹ sư Nghiên cứu Cấp cao & Kiến trúc sư Hệ thống Định vị Trong nhà (IPS) UWB của phòng lab.

KIẾN TRÚC 2 HỆ THỐNG THỰC NGHIỆM CỐT LÕI CỦA PHÒNG LAB (Thư mục System/):
1. Hệ thống 1 - Xe Tự Hành CAR (System/CAR/):
   • Phần cứng: Robot AGV STM32 + Raspberry Pi 4 + Decawave DWM1001C UWB Tag + 4 Anchor cố định (A0-A3).
   • Cổng Serial: /dev/ttyACM1 (921600 bps - bản tin CIR 59 trường) & /dev/ttyACM0 (115200 bps - Robot STM32 $IMU, Odom bánh xe).
   • Các script thuật toán: realtime_ai_lse.py (Tiny 1D-CNN Pure NumPy 57D bù sai số range bias trước khi giải LSE), realtime_fgoekf.py (EKF kết hợp Factor Graph Optimization cửa sổ trượt Huber Loss triệt tiêu NLOS), realtime_ekf.py (Dead Reckoning + UWB), realtime_positioning.py, realtime_lse.py.

2. Hệ thống 2 - Mạng UWB Đa Tag Vòng Token Ring (System/RING_Tag/):
   • Mục đích: Định vị đồng thời nhiều Tag (Multi-Tag tracking) không gây xung đột vô tuyến UWB (Collision Avoidance).
   • Nền tảng: Nordic nRF52832 + Decawave DW1000 chạy FreeRTOS trên DWM1001-DEV.
   • 3 loại Node: ss_bridge (Master điều phối toàn mạng, quản lý current_ring và current_anchor_list), ss_twr_init (Tag di động, chỉ đo TWR khi cầm Token TOKN 0xE2), ss_twr_resp (Anchor mốc phản hồi POLL/RESP).
   • Bộ khung bản tin: POLL (0xE0), RESP (0xE1), TOKN (0xE2), DATA (0xE3), RING (0xE4), AHLO (0xE5), ALST (0xE6), ARMV (0xE7), ADED (0xE8).
   • 5 cơ chế tự phục hồi lỗi (Self-Healing): (1) Tag Timeout 1000ms & rebuild ring, (2) Rhythm Disorder ngưỡng 50 -> phát lại RING 0xE4, (3) Dead Anchor Removal (2 chu kỳ liên tiếp 0xFFFFFFFF -> broadcast A_REMOVE & A_LIST), (4) Anchor Dynamic Re-join (nhận AHLO -> cập nhật A_LIST), (5) Resync Timers 50ms, Cooldown 10ms, Tag retry 5 lần.

• Phân hệ bổ trợ CIR Extraction (System/CIR_SYSTEM/): Trích xuất 1016 mẫu phức CIR qua thanh ghi SPI 0x25 ACC_MEM.

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. Khi người dùng hỏi hệ thống hiện tại gồm mấy hệ thống (hoặc trong repo có những hệ thống nào): BẮT BUỘC khẳng định ngay là phòng lab có 2 HỆ THỐNG THỰC NGHIỆM CỐT LÕI (Xe tự hành CAR và Mạng đa Tag Token Ring RING_Tag) cùng phân hệ bổ trợ CIR_SYSTEM. Kẻ Bảng Markdown chi tiết so sánh 2 hệ thống, TUYỆT ĐỐI KHÔNG hỏi ngược lại người dùng.
2. CIR trong UWB luôn là Channel Impulse Response (Đáp ứng Xung Kênh). Thanh ghi DW1000: 0x25 (ACC_MEM 1016 mẫu phức I, Q), 0x12:04 (FP_INDEX), 0x12:08 (FP_AMPLITUDE).
3. Khi phân tích, luôn đưa ra số liệu định lượng (RMSE, mm, cm, %) và dùng Bảng Markdown đẹp mắt (|:---:|:---|...).
4. Chỉ tập trung trả lời đúng trọng tâm câu hỏi hiện tại của người dùng, không lặp lại nội dung các lượt trước."""

        all_msgs = [{"role": "system", "content": effective_sys}] + messages


        payload = {
            "model": selected_model,
            "messages": all_msgs,
            "stream": True,
            "keep_alive": "60m",  # Giữ model trong RAM 60 phút, tránh độ trễ nạp lại từ ổ đĩa
            "options": {
                "num_ctx": 2048,       # Tối ưu tốc độ xử lý trên CPU Zen 3 (nhanh gấp 4 lần so với 8192)
                "temperature": 0.3,
                "repeat_penalty": 1.15 # Ngăn chặn triệt để hiện tượng lặp lại cụm từ hoặc cấu trúc câu cũ
            }
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    yield f"\n[Ollama {r.status_code} -> Chuyển sang AI dự phòng...]\n\n"
                    yield from self._stream_free_unlimited(system_prompt, messages)
                    return
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
        except Exception as e:
            yield f"\n[Lỗi kết nối Ollama: {str(e)} -> Chuyển sang AI dự phòng...]\n\n"
            yield from self._stream_free_unlimited(system_prompt, messages)

    def _stream_online(self, system_prompt: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Tự động phân luồng: Groq (gsk_...) hoặc Google Gemini (AIzaSy...) hoặc OpenAI (sk-...)."""
        if not self.api_key:
            yield "\n[Chưa có API Key -> Chuyển sang Kênh AI Dự phòng...]\n\n"
            yield from self._stream_free_unlimited(system_prompt, messages)
            return

        # 1. GROQ CLOUD (Hỗ trợ luân phiên 3 model + max_tokens an toàn chống lỗi 429)
        if self.api_key.startswith("gsk_"):
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            all_msgs = [{"role": "system", "content": system_prompt}] + messages
            
            groq_candidates = [
                ("openai/gpt-oss-120b", 1200),
                ("qwen/qwen3.8-27b", 800),
                ("openai/gpt-oss-20b", 1500)
            ]
            
            success = False
            for model_cand, max_tok in groq_candidates:
                payload = {
                    "model": model_cand,
                    "messages": all_msgs,
                    "stream": True,
                    "max_tokens": max_tok,
                    "temperature": 0.3
                }
                try:
                    r = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
                    if r.status_code == 200:
                        success = True
                        for line in r.iter_lines():
                            if line:
                                s = line.decode("utf-8")
                                if s.startswith("data: ") and s != "data: [DONE]":
                                    try:
                                        data = json.loads(s[6:])
                                        choices = data.get("choices", [])
                                        if choices:
                                            delta = choices[0]["delta"].get("content", "")
                                            if delta:
                                                yield delta
                                    except Exception:
                                        continue
                        break
                    elif r.status_code == 429:
                        continue
                    else:
                        continue
                except Exception:
                    continue

            if not success:
                yield "\n[⚠️ Tất cả model Groq đang chạm trần hạn mức phút -> Tự động chuyển sang Kênh AI Dự phòng...]\n\n"
                yield from self._stream_free_unlimited(system_prompt, messages)
            return

        # 2. GOOGLE GEMINI (Gemini 2.5 Flash / 1.5 Flash - Ngữ cảnh 1M tokens)
        model_name = self.model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={self.api_key}&alt=sse"

        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3}
        }

        try:
            with requests.post(url, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    yield f"\n[Lỗi Gemini API ({r.status_code}): {r.text}]\n"
                    return
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_json = line_str[6:].strip()
                            if data_json == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_json)
                                candidates = chunk.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts:
                                        if "text" in part:
                                            yield part["text"]
                            except Exception:
                                continue
        except Exception as e:
            yield f"\n[Lỗi Gemini API: {str(e)}]\n"
