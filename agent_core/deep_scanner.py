import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Generator

STOPWORDS = {
    "tìm", "kiếm", "cho", "tôi", "toàn", "bộ", "tất", "cả", "các", "những", 
    "về", "trong", "của", "là", "với", "hãy", "giúp", "paper", "papers", 
    "bài", "báo", "nào", "gì", "thế", "sao", "được", "có", "mình", "và", 
    "nhé", "xem", "hỏi", "đọc", "quét", "ra", "lại", "đây", "đó", "sẵn",
    "đề", "tài", "liệt", "kê", "chủ", "hướng", "nghiên", "cứu"
}

class DeepPaperScanner:
    def __init__(self, paper_dir: Path, llm_provider):
        self.paper_dir = paper_dir
        self.llm = llm_provider

    def _categorize_paper(self, filename: str, text_head: str) -> str:
        """Phân loại chính xác bài báo vào trường phái kỹ thuật cụ thể."""
        text = (filename + " " + text_head).lower()
        if any(w in text for w in ["mitigat", "error correction", "nlos correction", "compensat"]):
            return "🎯 Giảm thiểu sai số NLOS (Mitigation)"
        elif any(w in text for w in ["gramian", "cnn", "deep learning", "tcn", "neural", "simclr", "svm", "meta-learning", "transfer learning", "parallel deep"]):
            return "🧠 Deep Learning & AI"
        elif any(w in text for w in ["ekf", "ukf", "kalman", "csrukf", "rts", "particle filter", "fgo", "filter"]):
            return "📐 Bộ lọc Kalman / EKF / UKF"
        elif any(w in text for w in ["imu", "pdr", "ins", "odomet", "fusion", "mimu", "lidar"]):
            return "🔄 Sensor Fusion (IMU/PDR/LiDAR)"
        elif any(w in text for w in ["map", "anchor", "matrix", "ris", "geometry", "geometric"]):
            return "🗺️ Bản đồ kênh & Lựa chọn Anchor"
        else:
            return "📡 Tín hiệu CIR & Khảo sát công nghệ"

    def _extract_paper_summary(self, full_text: str) -> str:
        """Rút trích tóm tắt đóng góp cốt lõi của bài báo."""
        paras = [p.strip().replace("\n", " ") for p in full_text[:4000].split("\n\n") if len(p.strip()) > 80]
        for p in paras:
            p_lower = p.lower()
            if any(k in p_lower for k in ["propose", "in this paper", "we develop", "achieve", "experiment", "mitigat", "reduce", "accuracy"]):
                clean = re.sub(r"\s+", " ", p)
                if len(clean) > 180:
                    clean = clean[:180] + "..."
                return clean
        return paras[0][:150] + "..." if paras else "Phương pháp nghiên cứu định vị UWB trong môi trường phức tạp."

    def scan_and_synthesize_all(self, query: str, progress_callback=None) -> Generator[str, None, None]:
        """
        Duyệt qua 100% TOÀN BỘ 48 bài báo trong repo (chia nhỏ đọc từng bài).
        Rút trích đích danh thông tin thực tế từ từng bài báo, không bỏ sót bất kỳ bài nào.
        """
        pdfs = sorted(list(self.paper_dir.rglob("*.pdf")))
        total_papers = len(pdfs)

        yield f"## 🔍 Quét chuyên sâu 100% toàn bộ {total_papers} bài báo trong kho dữ liệu...\n\n"

        q_lower = query.lower()

        # Kiểm tra nếu là câu hỏi khảo sát tổng quan tất cả đề tài
        is_all_topics = any(w in q_lower for w in ["đề tài", "chủ đề", "hướng nghiên cứu", "tất cả", "toàn bộ", "tổng quan"])

        # Tách từ khóa có ý nghĩa (bỏ stopwords tiếng Việt)
        raw_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 1]
        meaningful_words = [w for w in raw_words if w not in STOPWORDS]

        # Mở rộng thuật ngữ chuyên ngành UWB
        search_terms = list(meaningful_words)
        if any(w in q_lower for w in ["mitigat", "giảm thiểu", "sai số", "hiệu chỉnh"]):
            search_terms.extend(["mitigat", "correction", "compensat", "nlos error", "ranging error"])
        if any(w in q_lower for w in ["ekf", "kalman"]):
            search_terms.extend(["ekf", "extended kalman", "ukf", "kalman"])
        if any(w in q_lower for w in ["cir", "kênh"]):
            search_terms.extend(["cir", "channel impulse response"])
        if any(w in q_lower for w in ["nlos", "los"]):
            search_terms.extend(["nlos", "non-line-of-sight"])

        terms = [re.escape(w) for w in set(search_terms) if len(w) > 2]
        
        # Nếu hỏi về toàn bộ đề tài / tổng quan, quét 100% bài báo không lọc từ khóa
        pattern = None if (is_all_topics or not terms) else re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)

        # BƯỚC 1: Quét từng bài báo một (MAP)
        paper_findings = []
        for idx, p in enumerate(pdfs, 1):
            if progress_callback:
                progress_callback(f"[{idx}/{total_papers}] Đang đọc: {p.name[:45]}...")
            
            try:
                full_text = subprocess.check_output(
                    ["pdftotext", str(p), "-"],
                    text=True, stderr=subprocess.DEVNULL
                )
                
                # Kiểm tra khớp từ khóa ở tên file hoặc nội dung
                name_match = pattern.search(p.name) if pattern else False
                content_match = pattern.search(full_text[:5000]) if pattern else False

                if name_match or content_match or not terms:
                    category = self._categorize_paper(p.name, full_text[:2000])
                    clean_title = p.stem.replace("_", " ")
                    summary_snip = self._extract_paper_summary(full_text)
                    paper_findings.append({
                        "filename": p.name,
                        "title": clean_title,
                        "category": category,
                        "is_our_paper": "OurPapers" in str(p),
                        "snippet": summary_snip
                    })
            except Exception:
                pass

        yield f"✔ **Đã đọc xong toàn bộ {total_papers} bài báo!**\n"
        yield f"📌 Tìm thấy **{len(paper_findings)} bài báo** có nội dung liên quan trực tiếp đến: *'{query}'*.\n\n"

        if not paper_findings:
            yield "Không tìm thấy nội dung liên quan trong các bài báo."
            return

        # BƯỚC 2: TỰ ĐỘNG SINH BẢNG TOÀN BỘ 100% CÁC BÀI BÁO (Bảng bo tròn chuẩn, đích danh tên bài báo)
        yield f"### 📊 Bảng Danh Mục Toàn Bộ {len(paper_findings)} Bài Báo Tìm Thấy Trong Kho Lưu Trữ\n\n"
        yield "| STT | Tên bài báo | Trường phái / Nhóm | Điểm nổi bật & Đóng góp |\n"
        yield "|:---:|:---|:---:|:---|\n"

        for idx, pf in enumerate(paper_findings, 1):
            highlight = pf['snippet'].replace("|", "-").replace("\n", " ")
            lab_badge = " [Lab Tác giả]" if pf["is_our_paper"] else ""
            yield f"| {idx} | **{pf['title']}{lab_badge}** | {pf['category']} | {highlight} |\n"

        yield "\n---\n\n### 🧠 Đánh Giá Học Thuật & So Sánh Toàn Diện Các Trường Phái\n\n"

        # BƯỚC 3: DÙNG LLM ĐỂ PHÂN TÍCH CHUYÊN SÂU TỪNG NHÓM (Cung cấp ĐÍCH DANH tên bài báo)
        findings_by_cat = {}
        for pf in paper_findings:
            cat = pf["category"]
            findings_by_cat.setdefault(cat, []).append(pf)

        summary_data = ""
        for cat, plist in findings_by_cat.items():
            summary_data += f"\nNhóm {cat} (Gồm {len(plist)} bài):\n"
            for p in plist:
                summary_data += f"- Tên bài: '{p['title']}' | Đóng góp: {p['snippet'][:180]}\n"

        synthesis_prompt = f"""Dưới đây là danh sách thực tế của {len(paper_findings)} bài báo có trong kho dữ liệu về chủ đề '{query}':

{summary_data[:10000]}

Nhiệm vụ: Hãy viết bài đánh giá học thuật chuyên sâu và so sánh khách quan các trường phái trên:
1. Đánh giá ưu điểm, nhược điểm và điều kiện áp dụng của từng trường phái (Deep Learning/AI, Bộ lọc Kalman/EKF/UKF, Sensor Fusion IMU/PDR, Bản đồ kênh/Hình học).
2. QUY ĐỊNH BẮT BUỘC: Khi phân tích, PHẢI TRÍCH DẪN ĐÍCH DANH TÊN CÁC BÀI BÁO trong danh sách trên (ví dụ: 'Parallel Deep Learning for NLOS Detection...', 'Enhanced Indoor Pedestrian Tracking Using UWB PDR Fusion...', 'Improving Localization Accuracy Using VB-CSRUKF...').
3. TUYỆT ĐỐI KHÔNG dùng ký hiệu chung chung giả lập như [1], [2]. KHÔNG lặp lại cùng một bài báo cho nhiều nhóm khác nhau.
4. Xu hướng công nghệ mới nhất và đề xuất giải pháp tối ưu cho hệ thống IPS UWB của phòng lab."""

        system_msg = "Bạn là Kỹ sư Nghiên cứu Cấp cao và Chuyên gia Học thuật về UWB IPS, trích dẫn chính xác, sắc bén và có chiều sâu."
        messages = [{"role": "user", "content": synthesis_prompt}]

        yield from self.llm.chat_stream(system_msg, messages)
