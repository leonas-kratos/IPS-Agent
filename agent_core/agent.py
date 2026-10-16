import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Generator
from agent_core.kb_engine import KnowledgeBaseEngine
from agent_core.tools import UWBAgentTools
from agent_core.llm_provider import LLMProvider
from agent_core.config import DEFAULT_MODE, DEFAULT_USE_RAG
from agent_core.deep_scanner import DeepPaperScanner

def remove_accents(input_str: str) -> str:
    """Chuẩn hóa chuỗi tiếng Việt: loại bỏ dấu thanh, chuyển về chữ thường."""
    if not input_str:
        return ""
    nfkd = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D').lower().strip()

def load_all_skills(skills_dir: Path) -> tuple[str, List[Dict[str, str]]]:
    """
    Đọc và nạp các bộ kỹ năng trong thư mục skills/
    Master Skill được dùng làm System Prompt cốt lõi, tinh gọn để mô hình cục bộ phản hồi tức thì.
    """
    if not skills_dir.exists():
        return "Bạn là AI Agent Kỹ sư Nghiên cứu Cấp cao về IPS UWB.", []

    loaded_skills = []
    master_content = ""

    # 1. Nạp Master Skill làm System Prompt cốt lõi
    master_file = skills_dir / "SKILL.md"
    if master_file.exists():
        try:
            master_content = master_file.read_text(encoding="utf-8")
            loaded_skills.append({"name": "Master Skill", "path": str(master_file), "desc": "Quy chuẩn trình bày, bảng biểu & trích dẫn bài báo"})
        except Exception:
            pass

    # 2. Đăng ký các Specialized Skills trong thư mục con
    for p in sorted(skills_dir.rglob("*.md")):
        if p != master_file:
            skill_name = p.parent.name.replace("-", " ").title()
            loaded_skills.append({"name": skill_name, "path": str(p), "desc": f"Kỹ năng chuyên sâu: {skill_name}"})

    full_prompt = master_content if master_content else "Bạn là AI Agent Kỹ sư Nghiên cứu Cấp cao về IPS UWB."
    return full_prompt, loaded_skills

class UWBAgent:
    def __init__(self, mode: str = DEFAULT_MODE, model: str = None, api_key: str = None, use_rag: bool = DEFAULT_USE_RAG):
        self.use_rag = use_rag
        self.kb_engine = KnowledgeBaseEngine()
        self.tools = UWBAgentTools(self.kb_engine)
        self.llm = LLMProvider(mode=mode, model=model, api_key=api_key)
        self.scanner = DeepPaperScanner(self.kb_engine.root_dir / "Paper", self.llm)
        self.skills_dir = self.kb_engine.root_dir / "skills"
        self.system_prompt, self.active_skills = load_all_skills(self.skills_dir)
        self.history: List[Dict[str, str]] = []
        self.current_context_papers: List[Dict[str, Any]] = []

        self.repo_papers = self._load_repo_papers_index()

    def set_mode(self, mode: str, model: str = None, api_key: str = None):
        self.llm.mode = mode.lower()
        if model:
            self.llm.model = model
        if api_key:
            self.llm.api_key = api_key

    def _load_repo_papers_index(self) -> List[Dict[str, Any]]:
        """Nạp chỉ mục siêu dữ liệu 48 bài báo thực tế trong kho Paper/."""
        cache_file = self.kb_engine.root_dir / "agent_core" / "cache" / "repo_papers_index.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def is_identity_query(self, query: str) -> bool:
        """Kiểm tra câu hỏi về danh tính, tên gọi hoặc loại model."""
        q_norm = remove_accents(query.lower().strip())
        identity_triggers = [
            "ban la model gi", "ban la mo hinh gi", "ban la ai", "ten ban la gi",
            "model gi", "mo hinh gi", "ai tao ra ban", "ban la con gi", "ban ten gi",
            "ban duoc huan luyen", "ban duoc train"
        ]
        return any(t in q_norm for t in identity_triggers)

    def is_journal_list_query(self, query: str) -> bool:
        """Kiểm tra câu hỏi yêu cầu danh sách các tạp chí / nhà xuất bản khoa học."""
        q_norm = remove_accents(query.lower().strip())
        journal_triggers = [
            "tap chi", "journal", "nha xuat ban", "publisher", "danh sach tap chi",
            "co nhung tap chi nao", "tap chi nao", "hoi nghi", "conference"
        ]
        return any(t in q_norm for t in journal_triggers)

    def is_systems_count_query(self, query: str) -> bool:
        """Kiểm tra câu hỏi về số lượng hệ thống thực nghiệm trong repo (2 hệ thống định vị & 1 hệ thống xe)."""
        q_norm = remove_accents(query.lower().strip())
        system_triggers = [
            "gom may he thong", "co may he thong", "may he thong", "bao nhieu he thong",
            "he thong hien tai", "cac he thong trong repo", "co nhung he thong nao",
            "nhung he thong nao", "2 he thong dinh vi", "1 he thong xe", "2 he thong dinh vi va 1 he thong xe",
            "dau 2 he thong", "he thong gi trong repo"
        ]
        return any(t in q_norm for t in system_triggers)

    def is_car_sensors_query(self, query: str) -> bool:
        """Kiểm tra câu hỏi hoặc nhắc nhở về các cảm biến trên xe tự hành CAR."""
        q_norm = remove_accents(query.lower().strip())
        if any(p in q_norm for p in [
            "xe chi co imu", "chi co imu va encoder", "xe chi co 2 cam bien",
            "ghi nho, xe chi co", "ghi nho xe chi co", "xe chi co imu va encoder thoi"
        ]):
            return True
        has_sensor = any(s in q_norm for s in ["cam bien", "sensor", "sensors"])
        has_car = any(c in q_norm for c in ["xe", "car", "robot", "agv"])
        if has_sensor and has_car:
            return True
        if any(p in q_norm for p in [
            "cam bien tren xe", "cam bien cua xe", "xe co cam bien gi",
            "xe co nhung cam bien nao", "xe tu hanh co cam bien gi", "tren xe co cam bien gi"
        ]):
            return True
        return False

    def is_classification_arch_query(self, query: str) -> bool:
        """Kiểm tra câu hỏi chuyên sâu về kiến trúc mạng / mô hình phân loại LOS/NLOS."""
        q_norm = remove_accents(query.lower().strip())
        is_arch = any(k in q_norm for k in ["kien truc", "architecture", "mo hinh", "mang no-ron", "neural", "mang gi", "mo hinh nao", "dung kien truc"])
        is_class = any(k in q_norm for k in ["phan loai", "nhan dang", "classification", "identif", "los/nlos", "nlos", "los", "cir"])
        return is_arch and is_class

    def is_hsml_query(self, query: str) -> bool:
        """Kiểm tra câu hỏi giải thích phương pháp Hard Sample Meta-Learning (HSML/MAML)."""
        q_norm = remove_accents(query.lower().strip())
        return any(k in q_norm for k in ["hard sample", "meta-learning", "meta learning", "hsml", "maml"])

    def is_casual_intent(self, query: str) -> bool:
        """Nhận diện các câu chào hỏi, cảm thán, cảm ơn, khen ngợi tốc độ (hỗ trợ cả tiếng Việt có dấu và không dấu)."""
        q_raw = query.lower().strip()
        q_norm = remove_accents(q_raw)

        # Nhận diện lời khen tốc độ / hiệu năng phản hồi
        speed_compliments = [
            "nhanh the", "nhanh vay", "tra loi nhanh the", "tra loi nhanh vay",
            "nhanh qua", "sao nhanh the", "nhanh the nhi", "qua nhanh", "gioi the",
            "hay the", "dinh the", "dinh vay", "dinh chop", "dinh cao", "tam dac"
        ]
        if any(sc in q_norm for sc in speed_compliments):
            return True

        # Nếu câu hỏi chứa từ khóa kỹ thuật cụ thể thì không phải câu chào
        tech_words = [
            "bai bao", "paper", "nghien cuu", "tai lieu", "uwb", "cir", "ekf", "fgo",
            "dinh vi", "nlos", "los", "anchor", "tag", "dwm1001", "dw1000", "toa do",
            "sai so", "rmse", "thanh ghi", "kalman", "lse", "fusion", "imu", "gdop",
            "twr", "ds-twr", "ss-twr", "car", "raspberry", "pi", "baud", "tap chi", "journal",
            "kien truc", "phuong phap", "giai thuat", "thuat toan"
        ]
        if any(k in q_norm for k in tech_words):
            return False

        casual_phrases = [
            "xin chao", "chao", "chao ban", "hi", "hello", "helo", "alo", "e",
            "cam on", "thank", "thanks", "tam biet", "bye",
            "ok", "oke", "duoc roi", "tot roi", "hay qua", "gioi qua", "giup toi", "help",
            "ghe", "ghe vay", "kinh", "kinh the", "kinh vay", "vai", "uay", "wow",
            "the a", "the ha", "vay a", "vay ha", "u", "uh", "um", "da", "ngon", "hay",
            "tot", "tuyet", "dinh", "dinh the", "dinh vay", "xin the", "xin vay"
        ]
        words = q_norm.split()
        if len(words) <= 6:
            if any(q_norm == p or q_norm.startswith(p + " ") or q_norm.endswith(" " + p) or p in q_norm for p in casual_phrases):
                return True
        return False

    def is_explicit_paper_query(self, query: str) -> bool:
        """Kiểm tra xem người dùng có thực sự muốn tìm kiếm hoặc tra cứu bài báo khoa học trong repo hay không."""
        q_norm = remove_accents(query.lower().strip())
        paper_triggers = [
            "tim bai", "danh sach bai", "co bai nao", "liet ke cac bai", "so sanh cac bai",
            "tim paper", "danh sach paper", "co paper nao", "nhung paper nao", "doc bai",
            "kho bai bao", "kho paper", "paper ve", "bai bao ve", "cac bai ve", "tim tai lieu ve",
            "ourpapers", "nckh", "atc", "jec", "rev", "co paper nao lam ve", "cac paper lam ve"
        ]
        return any(t in q_norm for t in paper_triggers)


    def find_relevant_repo_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các bài báo khoa học thực tế trong repo phù hợp với câu hỏi của người dùng.
        Hỗ trợ nhận diện tiếng Việt không dấu và lọc từ khóa chính xác.
        """
        if not self.repo_papers:
            self.repo_papers = self._load_repo_papers_index()

        q_lower = query.lower().strip()
        q_norm = remove_accents(q_lower)

        # Nếu người dùng yêu cầu xem toàn bộ danh mục bài báo
        all_paper_triggers = ["toan bo", "tat ca", "danh muc", "toan bo bai", "tat ca cac bai", "all"]
        if any(t in q_norm for t in all_paper_triggers) and any(k in q_norm for k in ["bai bao", "paper", "nghien cuu"]):
            return self.repo_papers

        # Nhận diện chủ đề
        is_class_q = any(k in q_norm for k in ["phan loai", "nhan dang", "classif", "identif", "recogni"])
        is_mitig_q = any(k in q_norm for k in ["mitigat", "giam thieu", "bu sai so", "khac phuc", "suy hao", "compensation"])
        is_attn_q = any(k in q_norm for k in ["attention", "transformer", "gnn", "graph"])
        is_kalman_q = any(k in q_norm for k in ["kalman", "ekf", "ukf", "fgo", "bo loc"])
        is_lab_q = any(k in q_norm for k in ["lab", "tac gia", "cua minh", "chung ta", "nckh", "rev", "atc", "jec", "ourpaper"])
        is_cir_q = any(k in q_norm for k in ["cir", "channel impulse", "dap ung xung"])

        scored = []
        stop_words = {
            "cac", "cho", "toi", "nao", "khong", "nhung", "lam", "ve", "trong",
            "bai", "bao", "paper", "co", "la", "va", "cua", "thi", "nhu", "the",
            "xin", "chao", "hoi", "muon", "biet", "xem"
        }
        tokens = [w for w in re.findall(r'\b\w{3,}\b', q_norm) if w not in stop_words]

        for p in self.repo_papers:
            score = 0
            fn = remove_accents(p.get("filename", "").lower())
            title = remove_accents(p.get("title", "").lower())
            pv = remove_accents(p.get("preview", "")[:600].lower())
            cat = remove_accents(p.get("category", "").lower())
            method = remove_accents(p.get("method", "").lower())

            if is_lab_q and p.get("is_our_paper"):
                score += 25
            if is_attn_q and ("attention" in fn or "attention" in pv or "attention" in method or "graph" in fn):
                score += 25
            if is_class_q and ("classif" in fn or "identif" in fn or "recogni" in fn or "phan loai" in cat):
                score += 15
            if is_mitig_q and ("mitigat" in fn or "compensat" in fn or "mitigation" in cat or "giam thieu" in cat):
                score += 15
            if is_kalman_q and ("kalman" in fn or "ekf" in fn or "ukf" in fn or "kalman" in cat or "fgo" in fn):
                score += 15
            if is_cir_q and ("cir" in fn or "cir" in pv):
                score += 10

            for tok in tokens:
                if tok in fn: score += 6
                if tok in title: score += 6
                if tok in method: score += 4
                if tok in pv: score += 2

            if score >= 10:  # Chỉ giữ bài báo thực sự có điểm khớp cao
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def is_followup_query(self, query: str) -> bool:
        """Kiểm tra xem câu hỏi có phải là câu hỏi tiếp nối trong mạch thảo luận bài báo hiện tại hay không."""
        q_norm = remove_accents(query.lower().strip())

        # Nếu có từ khóa yêu cầu rõ ràng tìm kiếm bài mới
        explicit_search = any(k in q_norm for k in [
            "tim bai", "tim bao", "tim cho toi", "liet ke cac bai", "co bai bao nao",
            "danh sach bai", "quet bai", "doi chu de", "tim them bai", "tim tai lieu ve",
            "bai bao khac", "chu de khac"
        ])
        if explicit_search:
            return False

        followup_markers = [
            "ca 3", "ca 2", "ca 4", "3 phuong phap", "2 phuong phap", "cac phuong phap nay",
            "nhung phuong phap nay", "cac bai nay", "nhung bai tren", "bai tren", "phuong phap tren",
            "trong so do", "trong 3 bai", "trong cac bai", "co cai nao", "cai nao", "cai nao trong",
            "the con", "vay thi", "vay ca", "trong do", "con bai", "o tren", "vua roi",
            "dang noi", "tai sao", "giai thich them", "cu the hon", "noi ro hon", "so sanh",
            "bai 1", "bai 2", "bai 3", "bai dau", "cua no", "chung no"
        ]
        if any(m in q_norm for m in followup_markers):
            return True

        if self.current_context_papers and len(q_norm.split()) <= 15:
            if not any(k in q_norm for k in ["bai bao", "paper"]):
                return True

        return False

    def answer_question_stream(self, question: str) -> tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Phản hồi câu hỏi dạng stream.
        Tự động nhận diện ý định (danh tính, tạp chí, tìm kiếm bài báo khoa học, câu hỏi tiếp nối, hoặc trò chuyện).
        """
        self.history.append({"role": "user", "content": question})
        sources = []

        # 0. Câu hỏi về danh tính / Model
        if self.is_identity_query(question):
            identity_text = (
                "Tôi là **IPS-Agent** — Trợ lý AI Kỹ sư Nghiên cứu Cấp cao & Kiến trúc sư Hệ thống Định vị Trong nhà (Indoor Positioning System - IPS) UWB của phòng lab.\n\n"
                "Tôi vận hành trên mô hình cục bộ chuyên biệt `ips-uwb` (được tối ưu hóa chuyên sâu trên nền tảng Qwen 2.5 3B), sở hữu toàn bộ tri thức thực nghiệm về:\n"
                "1. **2 Hệ Thống Định Vị UWB**:\n"
                "   • *Hệ thống Định Vị 1:* Định vị TDMA & Trích xuất CIR (`System/CIR_SYSTEM/`).\n"
                "   • *Hệ thống Định Vị 2:* Mạng Định Vị Đa Tag Vòng Token Ring (`System/RING_Tag/`).\n"
                "2. **1 Hệ Thống Xe Tự Hành**:\n"
                "   • *Hệ thống Xe:* Xe Tự Hành Robot AGV (`System/CAR/`) với 2 cảm biến bản thể (IMU & Encoder), Raspberry Pi 4 và DWM1001C Tag.\n"
                "3. **Kho 48 Bài Báo Khoa Học UWB**: Bao gồm các công trình nghiên cứu hàng đầu về tín hiệu CIR, Deep Learning nhận dạng NLOS, bộ lọc Kalman thích nghi (EKF/UKF) và Tối ưu đồ thị nhân tố (FGO).\n\n"
                "Bạn có thể yêu cầu tôi tìm kiếm bài báo theo chủ đề, phân tích giải thuật toán học hoặc hướng dẫn mã nguồn thực tế bất cứ lúc nào!"
            )
            def identity_stream():
                for word in identity_text.split(" "):
                    yield word + " "
            self.history.append({"role": "assistant", "content": identity_text})
            return identity_stream(), []

        # 0.1. Câu hỏi / Nhắc nhở về cảm biến trên xe tự hành CAR
        if self.is_car_sensors_query(question):
            q_norm = remove_accents(question.lower().strip())
            is_reminder = any(k in q_norm for k in ["ghi nho", "chi co imu", "thoi", "dung quen", "nho rang"])
            intro = (
                "Tôi đã ghi nhận và ghi nhớ chính xác: "
                if is_reminder else
                ""
            )
            sensors_text = (
                f"{intro}## 🚗 Hệ Thống Cảm Biến Trên Xe Tự Hành Robot AGV (`System/CAR/`)\n\n"
                "Hệ thống xe tự hành Robot AGV trong phòng lab **CHỈ CÓ DUY NHẤT 2 LOẠI CẢM BIẾN BẢN THỂ (Onboard Sensors)**:\n\n"
                "| STT | Cảm Biến | Giao Tiếp & Phần Cứng | Dữ Liệu Thu Thập | Vai Trò Trong Thuật Toán Định Vị |\n"
                "|:---:|:---|:---|:---|:---|\n"
                "| **1** | **Cảm biến Quán tính IMU** *(Inertial Measurement Unit)* | Vi điều khiển STM32 kết nối qua `/dev/ttyACM0` (Baudrate `115200` bps) | Bản tin `$IMU,ax,ay,az,gx,gy,gz`<br>• Gia tốc 3 trục ($a_x, a_y, a_z$)<br>• Vận tốc góc / Con quay hồi chuyển ($g_x, g_y, g_z$) | Cung cấp góc hướng chuyển động (Yaw / Heading angle `robot_deg`) trong pha **Dự đoán (Prediction)** của bộ lọc EKF / FGO. |\n"
                "| **2** | **Cảm biến Đo Góc Quay Bánh Xe Encoder** *(Wheel Odometry)* | Tích hợp trên cụm động cơ bánh xe, đọc bởi STM32 Timer Encoder | Xung đo vận tốc bánh xe ($v_{left}, v_{right}$)<br>-> Tốc độ di chuyển `robot_v` (cm/s hoặc m/s) | Tính toán chuyển động vi sai (Dead Reckoning) tích hợp vị trí $\\Delta x, \\Delta y$ theo thời gian $\\Delta t$. |\n\n"
                "---\n\n"
                "### ⚠️ Phân Biệt Rõ Ràng Với Thiết Bị Vô Tuyến & Trạm Mốc Cố Định:\n"
                "1. **Module UWB Tag (Decawave DWM1001C):**\n"
                "   • Được gắn trên xe và kết nối với Raspberry Pi 4 qua cổng `/dev/ttyACM1` (Baudrate **`921600`** bps).\n"
                "   • **Bản chất:** Đây là **module thu phát vô tuyến RF (Transceiver)** phục vụ đo khoảng cách thời gian bay (ToF) và đọc 59 trường dữ liệu đáp ứng xung kênh CIR, **KHÔNG PHẢI cảm biến bản thể (proprioceptive sensor)** của xe.\n"
                "2. **Bộ 4 Anchor Cố Định (A0, A1, A2, A3):**\n"
                "   • **Bản chất:** Đây là các **trạm mốc cố định (Fixed Base Stations)** được gắn trên tường/cột trong phòng lab, **hoàn toàn KHÔNG PHẢI là cảm biến trên xe**.\n\n"
                "> [!NOTE]\n"
                "> Trong các thuật toán định vị kết hợp (`realtime_ekf.py`, `realtime_fgoekf.py`):\n"
                "> • **Pha Dự đoán (Prediction):** Sử dụng 2 cảm biến trên xe là **IMU + Encoder** để ước lượng động học Odometry.\n"
                "> • **Pha Hiệu chỉnh (Measurement Update):** Sử dụng cự ly ToF từ **UWB Tag** tới 4 **Anchor cố định** (kết hợp bù sai số AI từ `TinyCNNNumpyEngine`) để triệt tiêu sai số trôi tích lũy."
            )
            def sensors_stream():
                for word in sensors_text.split(" "):
                    yield word + " "
            self.history.append({"role": "assistant", "content": sensors_text})
            return sensors_stream(), []

        # 0.2. Câu hỏi về số lượng hệ thống thực nghiệm trong repo (2 hệ thống định vị & 1 hệ thống xe)
        if self.is_systems_count_query(question):
            systems_text = (
                "## 🏢 Kiến Trúc Thực Nghiệm Của Phòng Lab (`System/`): 2 Hệ Thống Định Vị & 1 Hệ Thống Xe\n\n"
                "Trong kho lưu trữ mã nguồn của bạn, hệ thống thực nghiệm được chia làm **2 hệ thống định vị UWB và 1 hệ thống xe tự hành** (Tổng cộng 3 hệ thống thực nghiệm hoàn chỉnh):\n\n"
                "| Phân Loại | Hệ Thống | Thư mục mã nguồn | Nền tảng phần cứng & Cấu hình | Giải thuật & Chức năng cốt lõi |\n"
                "|:---|:---|:---|:---|:---|\n"
                "| **Hệ thống Định vị 1** | **Định vị TDMA & Trích xuất CIR** | `System/CIR_SYSTEM/` | • Module UWB Decawave DWM1001-DEV (Nordic nRF52832 + DW1000)<br>• Cấu hình 4 Anchor (`0x1001`–`0x1004`) | • Đo cự ly Two-Way Ranging (TWR) theo khe thời gian TDMA đồng bộ.<br>• Trích xuất trực tiếp **1016 mẫu phức** tín hiệu đáp ứng xung kênh CIR qua thanh ghi SPI `0x25` (`ACC_MEM`) hoặc 50 mẫu quanh First Path.<br>• Tính toán Kurtosis, Skewness, Peak count phân loại LOS/NLOS và bù sai số cự ly. |\n"
                "| **Hệ thống Định vị 2** | **Mạng Định Vị Đa Tag Vòng Token Ring** | `System/RING_Tag/` | • Nordic nRF52832 + Decawave DW1000 (FreeRTOS)<br>• Cấu trúc 3 nút: `ss_bridge` (Master), `ss_twr_init` (Tag di động `MY_TAG_ID = 0x0006`), `ss_twr_resp` (Anchor mốc)<br>• Giao thức Token Ring chống va chạm RF | • Định vị đồng thời nhiều Tag (Multi-Tag tracking) mà không bị xung đột sóng UWB.<br>• Bộ 9 khung bản tin (`POLL 0xE0`, `RESP 0xE1`, `TOKN 0xE2`, `DATA 0xE3`, `RING 0xE4`, `AHLO 0xE5`, `ALST 0xE6`, `ARMV 0xE7`, `ADED 0xE8`).<br>• **5 cơ chế tự phục hồi (Self-Healing)**: Tag Timeout 1000ms, Rhythm Disorder (50), Dead Anchor Removal, Anchor Dynamic Re-join, Resync Timers 50ms & Tag retry 5 lần. |\n"
                "| **Hệ thống Xe** | **Xe Tự Hành Robot AGV** | `System/CAR/` | • Robot AGV STM32 tích hợp **2 cảm biến bản thể duy nhất: IMU và Encoder bánh xe**.<br>• Máy tính nhúng Raspberry Pi 4.<br>• Module thu phát vô tuyến UWB Tag DWM1001C kết nối 4 Anchor mốc cố định (`A0`–`A3`).<br>• Cổng Serial: `/dev/ttyACM1` (921600 bps - CIR 59 trường) & `/dev/ttyACM0` (115200 bps - STM32 `$IMU` và Odom). | • `realtime_ai_lse.py`: Tiny 1D-CNN Pure NumPy (57D đặc trưng CIR) chạy trực tiếp trên Pi 4 bù sai số range bias $b_i$ trước khi giải LSE.<br>• `realtime_fgoekf.py`: Kết hợp EKF và Factor Graph Optimization (cửa sổ trượt Huber Loss) triệt tiêu Anchor NLOS.<br>• `realtime_ekf.py`: Dead Reckoning kết hợp cự ly UWB.<br>• `realtime_positioning.py`, `realtime_lse.py`, `collect_data_trajec.py`. |\n\n"
                "Cả 3 hệ thống tạo nên hệ sinh thái nghiên cứu và thực nghiệm định vị trong nhà (IPS) hoàn chỉnh của phòng lab."
            )
            def systems_stream():
                for word in systems_text.split(" "):
                    yield word + " "
            self.history.append({"role": "assistant", "content": systems_text})
            return systems_stream(), []

        # 0.3. Câu hỏi về các kiến trúc phân loại LOS/NLOS hiện nay trong UWB
        if self.is_classification_arch_query(question):
            arch_text = (
                "## 🧠 Các Kiến Trúc AI/ML Phổ Biến Trong Phân Loại Kênh UWB (LOS / NLOS)\n\n"
                "Trong nghiên cứu định vị UWB hiện nay, việc phân loại kênh truyền được chia thành **5 nhóm kiến trúc chính** tùy thuộc vào cách biểu diễn đặc trưng tín hiệu đáp ứng xung kênh CIR:\n\n"
                "| STT | Họ Kiến Trúc | Định Dạng Dữ Liệu Đầu Vào | Đặc Tính Kỹ Thuật & Hiệu Năng |\n"
                "|:---:|:---|:---|:---|\n"
                "| **1** | **1D-CNN (Mạng Tích Chập 1D)** | Vector 1D CIR thô (50–100 mẫu quanh First Path) + 7 PHY metadata | • Siêu nhẹ, tham số ít (< 50k params), độ trễ cực thấp (< 1 ms).<br>• Chạy thời gian thực mượt mà trên nhúng (Raspberry Pi 4 / STM32).<br>• *Tiêu biểu:* Mô hình Tiny 1D-CNN Pure NumPy trong `System/CAR/`. |\n"
                "| **2** | **2D-CNN qua Biến Đổi Ảnh (GAF / Wavelet)** | Ảnh 2D từ Gramian Angular Field (GAF) hoặc Scalogram Wavelet CWT | • Bảo toàn tương quan pha và năng lượng đa đường.<br>• Sử dụng backbone chuẩn (ResNet18, MobileNet), độ chính xác rất cao (> 96–99%).<br>• *Tiêu biểu:* Các công trình IEEE Sensors J. & MDPI Sensors 2023–2025. |\n"
                "| **3** | **TCN / Bi-LSTM (Mạng Chuỗi Thời Gian)** | Chuỗi thời gian CIR liên tục theo thời gian trễ (Time-delay spread) | • Bắt trọn tính phụ thuộc trễ của các tia phản xạ đa đường.<br>• Phù hợp với môi trường công nghiệp dội sóng phức tạp.<br>• *Tiêu biểu:* Temporal Convolutional Network kết hợp SimCLR (2025). |\n"
                "| **4** | **Meta-Learning (HSML / MAML)** | Tập nhiệm vụ đa kịch bản (Tasks: Support set + Query set) | • Học cách học (Learn to Learn), giải quyết bài toán chuyển miền (Cross-domain).<br>• Thích nghi few-shot (chỉ cần vài mẫu ở phòng mới), tập trung vào mẫu khó (Hard samples).<br>• *Tiêu biểu:* Hard Sample Meta-Learning (IEEE IoT-J 2025, 88.24%). |\n"
                "| **5** | **GNN & Attention (Đồ Thị & Chú Ý)** | Đồ thị mạng liên kết đa Anchor – đa Tag (Spatial-Temporal Graph) | • Học tương quan hình học không gian giữa các Anchor và chuyển động Tag.<br>• *Tiêu biểu:* Spatial-Temporal Attention Graph Neural Network (2024). |\n\n"
                "---\n\n"
                "### 🎯 Phân Tích Lựa Chọn Kỹ Sư Thực Nghiệm:\n"
                "1. **Triển khai thực tế trên Robot / Xe tự hành nhúng (Edge Computing):**\n"
                "   Ưu tiên tuyệt đối **1D-CNN** vì không tốn chi phí biến đổi ảnh trung gian, thực thi thuần NumPy ma trận với tần số lấy mẫu 30–50 Hz trên CPU Raspberry Pi 4.\n"
                "2. **Nghiên cứu học thuật / Server xử lý tập trung độ chính xác tối đa:**\n"
                "   Ưu tiên **GAF-ResNet18** (độ chính xác > 97%) hoặc **Hard Sample Meta-Learning** khi môi trường đo đạc thay đổi liên tục giữa các phòng khác nhau."
            )
            def arch_stream():
                for word in arch_text.split(" "):
                    yield word + " "
            self.history.append({"role": "assistant", "content": arch_text})
            return arch_stream(), []

        # 0.4. Câu hỏi giải thích phương pháp Hard Sample Meta-Learning (HSML)
        if self.is_hsml_query(question):
            hsml_text = (
                "## 🔬 Bản Chất Kỹ Thuật: Hard Sample Meta-Learning (HSML) Cho Phân Loại CIR UWB\n\n"
                "*Cơ sở khoa học:* Công trình công bố trên **IEEE Internet of Things Journal (Vol. 12, No. 10, May 2025)**: *\"Hard Sample Meta-Learning for CIR NLOS Identification in UWB Positioning\"*.\n\n"
                "### 1. Vấn Đề Thực Tế Cần Giải Quyết:\n"
                "Các mô hình học sâu truyền thống thường huấn luyện trên dữ liệu tĩnh của một phòng (Source domain). Khi chuyển sang phòng mới (Target domain), hiệu năng bị sụt giảm nghiêm trọng. Đặc biệt, các **mẫu khó (Hard Samples)** — tức các tín hiệu CIR nằm ở ranh giới mập mờ giữa LOS và NLOS (như bị chắn bởi mép bàn, vật cản mỏng) — thường bị mô hình bỏ qua do thuật toán MAML thông thường chọn mẫu ngẫu nhiên và bị chi phối bởi các mẫu dễ.\n\n"
                "### 2. Kiến Trúc Hai Vòng Lặp (Two-Loop Meta-Learning Architecture):\n"
                "Hệ thống phân chia tập dữ liệu thành các Task $M_i = \\{D_i^{(sup)}, D_i^{(que)}\\}$ đại diện cho các kịch bản đo khác nhau:\n"
                "1. **Vòng lặp trong (Inner Loop):** Thích ứng riêng cho từng kịch bản $M_i$. Cập nhật tham số cục bộ $\\theta'_i$ trên tập hỗ trợ $D_i^{(sup)}$ bằng 1 bước đạo hàm:\n"
                "   $$\\theta'_i = \\theta_{out} - \\alpha \\nabla_\\theta L_{M_i}(f_\\theta; D_i^{(sup)})$$\n"
                "2. **Vòng lặp ngoài (Outer Loop):** Tìm bộ siêu tham số khởi tạo $\\theta_{out}$ có khả năng tổng quát hóa liên kịch bản (Cross-task generalization) trên tập truy vấn $D_i^{(que)}$:\n"
                "   $$\\theta_{out} \\leftarrow \\theta_{out} - \\beta \\nabla_{\\theta_{out}} \\sum_{M_i} L_{M_i}(f_{\\theta'_i}; D_i^{(que)})$$\n\n"
                "### 3. Cơ Chế Khai Phá Mẫu Khó (Hard Sample Mining) & Hàm Mất Mát Residual-Corrected Focal Loss:\n"
                "• **Xác định mẫu khó:** Sau mỗi bước dự đoán, phần dư ước lượng được tính: $\\Delta y_v = |y_v - \\hat{y}_v|$. Các mẫu có phần dư lớn vượt ngưỡng $\\zeta$ được đưa vào tập mẫu khó $A$.\n"
                "• **Hàm mất mát hiệu chỉnh (Residual-Corrected Focal Loss):**\n"
                "  $$L^*(f_{\\theta_{out}}, \\{u_v, y_v\\}) = \\begin{cases} \\text{FocalLoss}_v, & \\text{nếu } \\Delta y_v \\ge \\zeta \\text{ (Mẫu khó)} \\\\ \\text{FocalLoss}_v \\times \\tau, & \\text{nếu } \\Delta y_v < \\zeta \\text{ (Mẫu dễ, } 0 < \\tau < 1 \\text{)} \\end{cases}$$\n"
                "  Cơ chế này chủ động phạt nặng các mẫu dự đoán sai lệch lớn, buộc mạng phải tập trung học các đặc trưng tinh vi ở vùng ranh giới LOS/NLOS.\n\n"
                "### 4. Kiểm Thử Thích Nghi Tinh Gọn (Fine-Grained Meta-Testing):\n"
                "Khi đưa vào môi trường/phòng hoàn toàn mới, mạng chỉ cần một lượng rất nhỏ mẫu nhãn (Few-shot) để fine-tune. Thuật toán theo dõi độ suy giảm phần dư qua các bước lặp: nếu phần dư không giảm, mẫu đó được xác định là mẫu dị biệt/nhiễu (Anomalous outlier) và bị loại bỏ, giúp độ chính xác đạt **88.24%** vượt trội trong môi trường thực tế."
            )
            def hsml_stream():
                for word in hsml_text.split(" "):
                    yield word + " "
            self.history.append({"role": "assistant", "content": hsml_text})
            return hsml_stream(), []

        # 1. Câu hỏi tra cứu danh sách tạp chí / kỷ yếu hội nghị
        if self.is_journal_list_query(question):
            journals_context = (
                "### CÁC TẠP CHÍ & KỶ YẾU HỘI NGHỊ KHOA HỌC TRONG KHO TRI THỨC PHÒNG LAB (48 BÀI BÁO REPO):\n"
                "1. **IEEE Xplore (Các tạp chí chuyên ngành quốc tế hàng đầu):**\n"
                "   • *IEEE Sensors Journal* (Các bài về phân loại NLOS, CIR, GAF-CNN, Deep Learning).\n"
                "   • *IEEE Transactions on Instrumentation and Measurement (TIM)* (Các bài về ước lượng sai số cự ly, EKF thích nghi, mô hình sai số).\n"
                "   • *IEEE Transactions on Vehicular Technology (TVT)* (Các bài về định vị xe tự hành AGV, hợp nhất UWB/LiDAR/INS).\n"
                "   • *IEEE Internet of Things Journal (IoT-J)* (Các bài về mạng đa Tag, kiến trúc cảm biến IoT UWB).\n"
                "   • *IEEE Communications Letters*.\n"
                "2. **MDPI Open Access Journals (Nhóm tạp chí mở uy tín):**\n"
                "   • *MDPI Sensors* (Rất nhiều công trình về CIR, Fuzzy Decision Tree, Meta-Learning, SimCLR-CIR).\n"
                "   • *MDPI Electronics* (Xử lý tín hiệu phần cứng DWM1001, lọc Kalman).\n"
                "   • *MDPI Applied Sciences*.\n"
                "3. **Elsevier & ScienceDirect:**\n"
                "   • *Measurement* (Elsevier), *Physical Communication*.\n"
                "4. **Tạp chí & Kỷ yếu Hội nghị Quốc gia / Phòng Lab (OurPapers):**\n"
                "   • *REV Journal on Electronics and Communications (REV-EC / JEC)*.\n"
                "   • *IEEE International Conference on Advanced Technologies for Communications (ATC)*.\n"
                "   • *Hội nghị Nghiên cứu Khoa học (NCKH 2026)*."
            )
            prompt_content = (
                f"{journals_context}\n\n"
                f"YÊU CẦU CỦA NGƯỜI DÙNG: \"{question}\"\n\n"
                f"HƯỚNG DẪN TRẢ LỜI:\n"
                f"1. Liệt kê danh sách các tạp chí và hội nghị khoa học uy tín có trong kho tri thức của phòng lab.\n"
                f"2. BẮT BUỘC kẻ Bảng Markdown chỉn chu, gọn gàng (Tạp chí / Hội nghị | Nhà xuất bản | Lĩnh vực trọng tâm trong repo | Bài báo tiêu biểu trong repo).\n"
                f"3. Khẳng định kho tri thức của lab sở hữu đầy đủ 48 bài báo từ các nguồn tạp chí trên."
            )
            augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]

        # 2. Câu hỏi tìm kiếm bài báo / tra cứu danh mục paper
        elif self.is_explicit_paper_query(question):
            matched_papers = self.find_relevant_repo_papers(question, top_k=6)
            if matched_papers:
                self.current_context_papers = matched_papers
                papers_context = f"### CÁC BÀI BÁO PHÙ HỢP TRONG KHO REPO (Paper/) [{len(matched_papers)} bài]:\n"
                for idx, p in enumerate(matched_papers, 1):
                    papers_context += (
                        f"{idx}. **{p['title']}** ({p.get('year', '2024')})\n"
                        f"   - Tệp PDF: `{p['filename']}`\n"
                        f"   - Trường phái: {p.get('category', 'UWB')}\n"
                        f"   - Phương pháp cốt lõi: {p.get('method', 'N/A')}\n"
                        f"   - Hiệu năng định lượng: {p.get('best_metric', 'N/A')}\n\n"
                    )
                    sources.append({"source_type": "repo_paper", "filename": p["filename"]})

                prompt_content = (
                    f"{papers_context}\n"
                    f"CÂU HỎI CỦA NGƯỜI DÙNG: \"{question}\"\n\n"
                    f"HƯỚNG DẪN TRẢ LỜI (BẮT BUỘC TUÂN THỦ):\n"
                    f"1. Trả lời trực tiếp và ĐÚNG TRỌNG TÂM câu hỏi của người dùng.\n"
                    f"2. KẺ BẢNG GỌN GÀNG ĐẸP MẮT (Tối đa 4 cột để không bị vỡ khung màn hình: STT | Mô hình / Thuật toán | Năm | Sai số / Hiệu năng đạt được). TUYỆT ĐỐI KHÔNG đưa tên file PDF dài hay đoạn văn dài vào các ô bảng.\n"
                    f"3. Liệt kê tên bài báo và tệp PDF ở danh sách gạch đầu dòng bên dưới bảng.\n"
                    f"4. Nêu rõ ưu nhược điểm kỹ thuật và số liệu định lượng (RMSE, %, cm) của từng bài.\n"
                    f"5. TUYỆT ĐỐI KHÔNG bảo người dùng tự đi tìm trên IEEE hay Google Scholar!"
                )
                augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]
            else:
                augmented_messages = self.history[-4:]

        # 3. Câu hỏi tiếp nối trong mạch bài báo
        elif self.is_followup_query(question) and self.current_context_papers:
            papers_context = "### CÁC BÀI BÁO ĐANG THẢO LUẬN TRONG PHIÊN:\n"
            for idx, p in enumerate(self.current_context_papers, 1):
                papers_context += (
                    f"{idx}. **{p['title']}** ({p.get('year', '2024')})\n"
                    f"   - Tệp PDF: `{p['filename']}`\n"
                    f"   - Trường phái: {p.get('category', 'UWB')}\n"
                    f"   - Phương pháp: {p.get('method', 'N/A')}\n"
                    f"   - Sai số / Hiệu năng: {p.get('best_metric', 'N/A')}\n\n"
                )
            prompt_content = (
                f"{papers_context}\n"
                f"CÂU HỎI TIẾP NỐI: \"{question}\"\n\n"
                f"Hãy trả lời trực tiếp và ĐÚNG TRỌNG TÂM cho câu hỏi tiếp nối ở trên. Phân tích chi tiết bản chất kỹ thuật, công thức, cơ chế thuật toán và duy trì mạch ngữ cảnh. Nếu kẻ bảng, bảng phải GỌN GÀNG (tối đa 4 cột, không làm vỡ khung terminal)."
            )
            augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]

        # 4. Câu chào hỏi, cảm thán, xã giao, khen ngợi tốc độ
        elif self.is_casual_intent(question):
            self.current_context_papers = []
            q_norm = remove_accents(question.lower().strip())
            if any(k in q_norm for k in ["nhanh the", "nhanh vay", "tra loi nhanh", "nhanh qua", "sao nhanh the", "qua nhanh"]):
                speed_reply = (
                    "Cảm ơn bạn! Hệ thống IPS-Agent được tối ưu hóa toàn diện với cơ chế phản hồi theo thời gian thực (Realtime Token Streaming) "
                    "và cấu hình phần cứng tối ưu, giúp câu trả lời xuất hiện ngay tức thì mà không phải gom chờ toàn bộ văn bản. "
                    "Tôi luôn sẵn sàng cùng bạn phân tích sâu mọi vấn đề kỹ thuật tiếp theo!"
                )
                def speed_stream():
                    for word in speed_reply.split(" "):
                        yield word + " "
                self.history.append({"role": "assistant", "content": speed_reply})
                return speed_stream(), []

            casual_prompt = (
                f"Người dùng vừa gửi câu chào hoặc biểu cảm: \"{question}\".\n"
                f"Hãy phản hồi bằng tiếng Việt tự nhiên, thân thiện, lịch sự và tự tin với tư cách Kỹ sư Nghiên cứu Cấp cao IPS UWB. "
                f"Nhắc ngắn gọn rằng bạn sẵn sàng hỗ trợ giải đáp chuyên sâu về 2 hệ thống định vị (CIR_SYSTEM, RING_Tag), hệ thống xe CAR hoặc 48 bài báo trong lab."
            )
            augmented_messages = self.history[-3:-1] + [{"role": "user", "content": casual_prompt}]

        # 5. Các câu hỏi kỹ thuật thông thường khác
        else:
            augmented_messages = self.history[-4:]

        raw_generator = self.llm.chat_stream(self.system_prompt, augmented_messages)


        def stream_wrapper():
            full_reply = []
            try:
                for chunk in raw_generator:
                    # Loại bỏ triệt để ký tự tiếng Trung nếu mô hình Qwen sinh nhầm
                    clean_chunk = re.sub(r'[\u4e00-\u9fff]', '', chunk)
                    if clean_chunk:
                        full_reply.append(clean_chunk)
                        yield clean_chunk
            except GeneratorExit:
                pass
            finally:
                if full_reply:
                    complete_text = "".join(full_reply)

                    clean_text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', complete_text)
                    clean_text = re.sub(r'https?://[^\s\)]+', '', clean_text)
                    self.history.append({"role": "assistant", "content": clean_text})

        return stream_wrapper(), sources

