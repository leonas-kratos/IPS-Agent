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

    def is_casual_intent(self, query: str) -> bool:
        """Nhận diện các câu chào hỏi, cảm ơn, xã giao (hỗ trợ cả tiếng Việt có dấu và không dấu)."""
        q_raw = query.lower().strip()
        q_norm = remove_accents(q_raw)

        # Nếu câu hỏi chứa từ khóa kỹ thuật cụ thể thì không phải câu chào
        tech_words = [
            "bai bao", "paper", "nghien cuu", "tai lieu", "uwb", "cir", "ekf", "fgo",
            "dinh vi", "nlos", "los", "anchor", "tag", "dwm1001", "dw1000", "toa do",
            "sai so", "rmse", "thanh ghi", "kalman", "lse", "fusion", "imu", "gdop",
            "twr", "ds-twr", "ss-twr", "car", "raspberry", "pi", "baud"
        ]
        if any(k in q_norm for k in tech_words):
            return False

        casual_phrases = [
            "xin chao", "chao", "chao ban", "hi", "hello", "helo", "alo", "e",
            "ban la ai", "ai the", "cam on", "thank", "thanks", "tam biet", "bye",
            "ok", "oke", "duoc roi", "tot roi", "hay qua", "gioi qua", "giup toi", "help"
        ]
        words = q_norm.split()
        if len(words) <= 4:
            if any(q_norm == p or q_norm.startswith(p + " ") or q_norm.endswith(" " + p) for p in casual_phrases):
                return True
        return False

    def is_explicit_paper_query(self, query: str) -> bool:
        """Kiểm tra xem người dùng có thực sự muốn tìm kiếm hoặc tra cứu bài báo khoa học hay không."""
        q_norm = remove_accents(query.lower().strip())
        paper_triggers = [
            "bai bao", "paper", "nghien cuu", "tai lieu", "cong bo", "tac gia",
            "tim bai", "danh sach bai", "co bai nao", "liet ke cac bai", "so sanh cac bai",
            "ourpapers", "nckh", "atc", "jec", "rev", "doc bai", "kho bai bao"
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
        - Với câu hỏi chào hỏi/xã giao: Phản hồi tự nhiên, thân thiện.
        - Với câu hỏi kỹ thuật/giải thuật/phần cứng: Sử dụng toàn bộ tri thức nén của repo trong mô hình.
        - Với câu hỏi tra cứu bài báo: Tra cứu chính xác và cung cấp ngữ cảnh bài báo thực tế.
        """
        self.history.append({"role": "user", "content": question})
        sources = []

        # 1. Nếu là câu chào hỏi / xã giao (có dấu hoặc không dấu)
        if self.is_casual_intent(question):
            self.current_context_papers = []
            augmented_messages = self.history[-4:]

        # 2. Nếu là câu hỏi tiếp nối mạch bài báo trước đó
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
                f"Hãy trả lời trực tiếp cho các bài báo đang thảo luận ở trên, phân tích chi tiết và duy trì mạch ngữ cảnh."
            )
            augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]

        # 3. Nếu là câu hỏi tra cứu tài liệu / bài báo khoa học cụ thể
        elif self.is_explicit_paper_query(question):
            matched_papers = self.find_relevant_repo_papers(question, top_k=5)
            if matched_papers:
                self.current_context_papers = matched_papers
                papers_context = f"### CÁC BÀI BÁO LIÊN QUAN TRONG KHO REPO (Paper/) [{len(matched_papers)} bài]:\n"
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
                    f"YÊU CẦU CỦA NGƯỜI DÙNG: \"{question}\"\n\n"
                    f"Hãy trả lời yêu cầu của người dùng một cách rõ ràng, trích dẫn chính xác bài báo và số liệu từ danh sách trên."
                )
                augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]
            else:
                augmented_messages = self.history[-4:]

        # 4. Các câu hỏi kỹ thuật, toán học, mã nguồn, phần cứng thông thường
        else:
            # Cho phép mô hình sử dụng toàn bộ tri thức của nó một cách tự do
            augmented_messages = self.history[-4:]

        raw_generator = self.llm.chat_stream(self.system_prompt, augmented_messages)

        def stream_wrapper():
            full_reply = []
            for chunk in raw_generator:
                full_reply.append(chunk)
                yield chunk
            if full_reply:
                complete_text = "".join(full_reply)
                clean_text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', complete_text)
                clean_text = re.sub(r'https?://[^\s\)]+', '', clean_text)
                self.history.append({"role": "assistant", "content": clean_text})

        return stream_wrapper(), sources
