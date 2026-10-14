import os
import re
from pathlib import Path
from typing import List, Dict, Any, Generator
from agent_core.kb_engine import KnowledgeBaseEngine
from agent_core.tools import UWBAgentTools
from agent_core.llm_provider import LLMProvider
from agent_core.config import DEFAULT_MODE, DEFAULT_USE_RAG
from agent_core.deep_scanner import DeepPaperScanner

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

    def find_relevant_repo_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các bài báo khoa học thực tế trong repo phù hợp với câu hỏi của người dùng.
        Hỗ trợ nhận diện các chủ đề cốt lõi: phân loại LOS/NLOS, CIR, giảm thiểu sai số, attention, Kalman, lab papers...
        """
        if not self.repo_papers:
            self.repo_papers = self._load_repo_papers_index()

        q_lower = query.lower().strip()
        
        # Bỏ qua các câu chào hỏi xã giao hoặc ngắn không mang tính tra cứu tài liệu
        casual_greetings = ["xin chào", "hello", "hi", "chào bạn", "bạn là ai", "cảm ơn", "tạm biệt"]
        if any(q_lower == g or q_lower.startswith(g + " ") for g in casual_greetings) and not any(k in q_lower for k in ["bài báo", "paper", "nghiên cứu", "tài liệu"]):
            return []

        # Các nhóm chủ đề quan trọng
        is_class_q = any(k in q_lower for k in ["phân loại", "nhận dạng", "classif", "identif", "recogni"])
        is_mitig_q = any(k in q_lower for k in ["mitigat", "giảm thiểu", "bù sai số", "khắc phục", "suy hao"])
        is_attn_q = any(k in q_lower for k in ["attention", "self attention", "self-attention", "transformer"])
        is_kalman_q = any(k in q_lower for k in ["kalman", "ekf", "ukf", "bộ lọc"])
        is_lab_q = any(k in q_lower for k in ["lab", "tác giả", "của mình", "chúng ta", "nckh", "rev", "atc", "jec"])
        is_cir_q = any(k in q_lower for k in ["cir", "channel impulse", "đáp ứng xung"])

        scored = []
        for p in self.repo_papers:
            score = 0
            fn = p.get("filename", "").lower()
            title = p.get("title", "").lower()
            pv = p.get("preview", "").lower()
            cat = p.get("category", "").lower()
            method = p.get("method", "").lower()

            if is_lab_q and p.get("is_our_paper"):
                score += 20
            if is_attn_q and ("attention" in fn or "attention" in pv or "attention" in method):
                score += 25
            if is_class_q and ("classif" in fn or "identif" in fn or "recogni" in fn or "phân loại" in cat):
                score += 12
            if is_mitig_q and ("mitigat" in fn or "compensat" in fn or "mitigation" in cat):
                score += 10
            if is_kalman_q and ("kalman" in fn or "ekf" in fn or "ukf" in fn or "kalman" in cat):
                score += 10
            if is_cir_q and ("cir" in fn or "cir" in pv[:500]):
                score += 5

            # So khớp từ khóa trực tiếp
            words = re.findall(r'\w{3,}', q_lower)
            for w in words:
                if w in ["các", "cho", "tôi", "nào", "không", "những", "làm", "về", "trong", "bài", "báo", "paper", "có"]:
                    continue
                if w in fn: score += 4
                if w in title: score += 4
                if w in method: score += 3
                if w in pv[:500]: score += 1

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def is_followup_query(self, query: str) -> bool:
        """Kiểm tra xem câu hỏi có phải là câu hỏi tiếp nối trong mạch thảo luận hiện tại hay không."""
        q = query.lower().strip()

        # Nếu có từ khóa yêu cầu rõ ràng tìm kiếm bài mới
        explicit_search = any(k in q for k in [
            "tìm bài", "tìm báo", "tìm cho tôi", "liệt kê các bài", "có bài báo nào",
            "danh sách bài", "quét bài", "đổi chủ đề", "tìm thêm bài", "tìm tài liệu về",
            "bài báo khác", "chủ đề khác"
        ])
        if explicit_search:
            return False

        # Các dấu hiệu của câu hỏi tiếp nối mạch truyện
        followup_markers = [
            "cả 3", "cả 2", "cả 4", "3 phương pháp", "2 phương pháp", "các phương pháp này",
            "những phương pháp này", "các bài này", "những bài trên", "bài trên", "phương pháp trên",
            "trong số đó", "trong 3 bài", "trong các bài", "có cái nào", "cái nào", "cái nào trong",
            "thế còn", "vậy thì", "vậy cả", "trong đó", "còn bài", "ở trên", "vừa rồi",
            "đang nói", "nhảy luôn", "tại sao", "giải thích thêm", "cụ thể hơn",
            "nói rõ hơn", "so sánh", "bài 1", "bài 2", "bài 3", "bài đầu", "của nó", "chúng nó"
        ]
        if any(m in q for m in followup_markers):
            return True

        # Nếu đang có bài báo trong ngữ cảnh và câu hỏi ngắn mang tính hỏi sâu
        if self.current_context_papers and len(q.split()) <= 15:
            if not any(k in q for k in ["bài báo", "paper"]):
                return True

        return False

    def answer_question_stream(self, question: str) -> tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Phản hồi câu hỏi dạng stream dựa 100% trên các bài báo thực tế trong repo (Paper/)
        và tri thức mô hình cục bộ. Giữ vững mạch hội thoại tiếp nối, không nhảy chủ đề bừa bãi.
        """
        self.history.append({"role": "user", "content": question})

        sources = []
        is_followup = self.is_followup_query(question)

        # 1. Nếu là câu hỏi tiếp nối và đang có danh sách bài báo trong ngữ cảnh
        if is_followup and self.current_context_papers:
            papers_context = "### CÁC BÀI BÁO ĐANG ĐƯỢC THẢO LUẬN TRONG PHIÊN HỘI THOẠI NÀY:\n"
            for idx, p in enumerate(self.current_context_papers, 1):
                is_imu_fusion = "Modified_Matrix" in p["filename"] or "imu" in p.get("title", "").lower()
                papers_context += (
                    f"{idx}. **{p['title']}** ({p['year']})\n"
                    f"   - Tệp PDF: `{p['filename']}`\n"
                    f"   - Trường phái: {p.get('category', 'UWB')}\n"
                    f"   - Phương pháp cốt lõi: {p.get('method', 'N/A')}\n"
                    f"   - Tín hiệu CIR: {'Có sử dụng CIR' if p.get('uses_cir') else 'KHÔNG dùng CIR (dùng khoảng cách ranging/ToF)'}\n"
                    f"   - Hợp nhất cảm biến (Sensor Fusion): {'CÓ (Hợp nhất UWB và IMU)' if is_imu_fusion else 'KHÔNG (Thuần UWB)'}\n"
                    f"   - Hiệu năng định lượng: {p.get('best_metric', 'N/A')}\n\n"
                )

            prompt_content = (
                f"{papers_context}\n"
                f"CÂU HỎI TIẾP NỐI CỦA NGƯỜI DÙNG: \"{question}\"\n\n"
                f"YÊU CẦU QUAN TRỌNG:\n"
                f"1. DUY TRÌ ngữ cảnh và trả lời TRỰC TIẾP cho các bài báo đang thảo luận ở trên.\n"
                f"2. TUYỆT ĐỐI KHÔNG tự ý nhảy sang bài báo hoặc tài liệu khác ngoài danh sách đang thảo luận.\n"
                f"3. Phân tích cụ thể, rõ ràng từng bài báo trong số các bài trên đối với câu hỏi người dùng."
            )
            augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]

        # 2. Nếu là tìm kiếm bài mới hoặc mở chủ đề mới
        else:
            matched_papers = self.find_relevant_repo_papers(question, top_k=3)
            if matched_papers:
                self.current_context_papers = matched_papers
                papers_context = "### CÁC BÀI BÁO KHOA HỌC THỰC TẾ TRONG KHO REPO (Paper/):\n"
                for idx, p in enumerate(matched_papers, 1):
                    papers_context += (
                        f"{idx}. **{p['title']}** ({p['year']})\n"
                        f"   - Tệp PDF: `{p['filename']}`\n"
                        f"   - Trường phái: {p.get('category', 'UWB')}\n"
                        f"   - Phương pháp cốt lõi: {p.get('method', 'N/A')}\n"
                        f"   - Hiệu năng định lượng: {p.get('best_metric', 'N/A')}\n"
                        f"   - Tạp chí/Kỷ yếu: {p.get('journal', 'Khoa học UWB')}\n\n"
                    )
                    sources.append({"source_type": "repo_paper", "filename": p["filename"]})

                prompt_content = (
                    f"{papers_context}\n"
                    f"CÂU HỎI CỦA NGƯỜI DÙNG: \"{question}\"\n\n"
                    f"YÊU CẦU BẮT BUỘC:\n"
                    f"1. CHỈ SỬ DỤNG thông tin từ các bài báo thực tế trong danh sách trên từ repo. Tuyệt đối KHÔNG bịa ra link hoặc bài báo ngoài danh sách.\n"
                    f"2. Lập BẢNG SO SÁNH Markdown chuẩn (các cột: STT, Tên bài báo, Năm, Phương pháp/Kiến trúc, Hiệu năng/Sai số, Tệp PDF).\n"
                    f"3. Cột Tệp PDF: Chỉ ghi tên tệp mã thô dạng `ten_tep.pdf`, TUYỆT ĐỐI KHÔNG chèn link http/https hay markdown link.\n"
                    f"4. Phân tích súc tích, ngắn gọn ưu điểm và cơ chế của các phương pháp trên."
                )
                augmented_messages = self.history[-3:-1] + [{"role": "user", "content": prompt_content}]
            else:
                augmented_messages = self.history[-4:]

        raw_generator = self.llm.chat_stream(self.system_prompt, augmented_messages)

        def stream_wrapper():
            full_reply = []
            for chunk in raw_generator:
                full_reply.append(chunk)
                yield chunk
            if full_reply:
                complete_text = "".join(full_reply)
                # Loại bỏ hoàn toàn bất kỳ link URL ảo nào nếu LLM vô tình sinh ra
                clean_text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', complete_text)
                clean_text = re.sub(r'https?://[^\s\)]+', '', clean_text)
                self.history.append({"role": "assistant", "content": clean_text})

        return stream_wrapper(), sources
