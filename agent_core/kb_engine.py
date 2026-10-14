import os
import re
import json
import gzip
import subprocess
from pathlib import Path
from typing import List, Dict, Any

CACHE_FILE = Path(__file__).parent / "cache" / "knowledge_base.json.gz"
CATALOG_FILE = Path(__file__).parent / "cache" / "paper_catalog.json"

class KnowledgeBaseEngine:
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent
        self.documents: List[Dict[str, Any]] = []
        self.paper_catalog: List[Dict[str, Any]] = []
        self._load_or_build_index()

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Trích xuất text từ file PDF sử dụng pdftotext."""
        try:
            res = subprocess.run(
                ["pdftotext", str(pdf_path), "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True
            )
            return res.stdout
        except Exception:
            return ""

    def _chunk_text(self, text: str, chunk_size: int = 1400, overlap: int = 250) -> List[str]:
        """Chia văn bản thành các đoạn có gối đầu (overlap)."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = " ".join(lines)
        if len(cleaned) <= chunk_size:
            return [cleaned] if cleaned else []
        
        chunks = []
        start = 0
        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            if end < len(cleaned):
                last_space = cleaned.rfind(" ", start, end)
                if last_space > start + chunk_size // 2:
                    end = last_space
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(cleaned) else len(cleaned)
        return chunks

    def _load_or_build_index(self):
        """Tải index và catalog từ cache."""
        loaded = False
        if CACHE_FILE.exists():
            try:
                with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as f:
                    self.documents = json.load(f)
                    loaded = True
            except Exception:
                pass

        if CATALOG_FILE.exists():
            try:
                with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                    self.paper_catalog = json.load(f)
            except Exception:
                pass

        if not loaded or not self.paper_catalog:
            self.rebuild_index()

    def rebuild_index(self, verbose_callback=None):
        """Quét toàn bộ thư mục Document, Paper, System và tạo index + catalog."""
        self.documents = []
        self.paper_catalog = []

        # 1. Quét Document (Datasheet, User Manual DWM1001)
        doc_dir = self.root_dir / "Document"
        if doc_dir.exists():
            for pdf_file in doc_dir.glob("*.pdf"):
                raw_text = self._extract_text_from_pdf(pdf_file)
                chunks = self._chunk_text(raw_text)
                for idx, chunk in enumerate(chunks):
                    self.documents.append({
                        "id": f"doc_{pdf_file.stem}_{idx}",
                        "source_type": "hardware_doc",
                        "filename": pdf_file.name,
                        "relative_path": str(pdf_file.relative_to(self.root_dir)),
                        "chunk_index": idx,
                        "text": chunk
                    })

        # 2. Quét Paper (Các bài báo khoa học UWB)
        paper_dir = self.root_dir / "Paper"
        if paper_dir.exists():
            for pdf_file in sorted(paper_dir.rglob("*.pdf")):
                raw_text = self._extract_text_from_pdf(pdf_file)
                chunks = self._chunk_text(raw_text)
                for idx, chunk in enumerate(chunks):
                    self.documents.append({
                        "id": f"paper_{pdf_file.stem}_{idx}",
                        "source_type": "research_paper",
                        "filename": pdf_file.name,
                        "relative_path": str(pdf_file.relative_to(self.root_dir)),
                        "chunk_index": idx,
                        "text": chunk
                    })
                
                # Thêm vào Paper Catalog tổng quan
                is_our_paper = "OurPapers" in str(pdf_file)
                first_page = raw_text[:800].replace("\n", " ") if raw_text else ""
                self.paper_catalog.append({
                    "filename": pdf_file.name,
                    "relative_path": str(pdf_file.relative_to(self.root_dir)),
                    "is_our_paper": is_our_paper,
                    "title": pdf_file.stem.replace("_", " "),
                    "preview": first_page[:250]
                })

        # 3. Quét System (Mã nguồn thực tế CAR, CIR, RING_Tag)
        sys_dir = self.root_dir / "System"
        if sys_dir.exists():
            for code_file in sys_dir.rglob("*.py"):
                if "__pycache__" in str(code_file):
                    continue
                try:
                    code_text = code_file.read_text(encoding="utf-8", errors="ignore")
                    chunks = self._chunk_text(code_text, chunk_size=1500, overlap=150)
                    for idx, chunk in enumerate(chunks):
                        self.documents.append({
                            "id": f"code_{code_file.stem}_{idx}",
                            "source_type": "system_code",
                            "filename": code_file.name,
                            "relative_path": str(code_file.relative_to(self.root_dir)),
                            "chunk_index": idx,
                            "text": chunk
                        })
                except Exception:
                    pass

        # Lưu cache
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE_FILE, "wt", encoding="utf-8") as f:
            json.dump(self.documents, f)
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.paper_catalog, f, ensure_ascii=False, indent=2)

    def get_catalog_summary(self) -> str:
        """Tạo bản đồ danh mục toàn bộ các bài báo trong repo."""
        if not self.paper_catalog:
            return ""
        
        our_papers = [p for p in self.paper_catalog if p.get("is_our_paper")]
        ref_papers = [p for p in self.paper_catalog if not p.get("is_our_paper")]

        output = ["=== MỤC LỤC TẤT CẢ CÁC BÀI BÁO TRONG KHO DỮ LIỆU ==="]
        if our_papers:
            output.append("\n📌 Nghiên cứu của nhóm tác giả (Paper/OurPapers/):")
            for p in our_papers:
                output.append(f"- `{p['filename']}`: {p['title']}")
        
        output.append(f"\n📚 Hơn {len(ref_papers)} bài báo khoa học tham khảo (Paper/):")
        for p in ref_papers:
            output.append(f"- `{p['filename']}`: {p['title']}")
        output.append("=== HẾT MỤC LỤC ===\n")
        return "\n".join(output)

    def search(self, query: str, source_type: str = None, top_k: int = 8, max_per_doc: int = 2) -> List[Dict[str, Any]]:
        """
        Tìm kiếm tài liệu với cơ chế Đa dạng hóa nguồn (Diversity Sampling):
        Không để 1 file chiếm hết kết quả, mỗi tài liệu tối đa `max_per_doc` đoạn.
        """
        if not self.documents:
            return []

        terms = [re.escape(w.lower()) for w in re.findall(r"\w+", query) if len(w) > 1]
        if not terms:
            return []

        keyword_weights = {
            "cir": 3.0, "nlos": 3.0, "los": 2.5, "ekf": 2.5, "ukf": 2.5,
            "fgo": 2.5, "dwm1001": 2.5, "dwm1001c": 2.5, "tlv": 2.5,
            "uart": 2.0, "spi": 2.0, "tag": 2.0, "anchor": 2.0,
            "ranging": 2.0, "twr": 2.5, "tof": 2.0, "tdoa": 2.0,
            "cnn": 2.0, "mlp": 2.0, "imu": 2.5, "fusion": 2.5,
            "car": 2.0, "lse": 2.0, "odometry": 2.5, "slam": 2.5
        }

        pattern = re.compile(r"\b(" + "|".join(terms) + r")\b", re.IGNORECASE)

        scored = []
        for doc in self.documents:
            if source_type and doc.get("source_type") != source_type:
                continue

            text_lower = doc["text"].lower()
            fname_lower = doc["filename"].lower()
            score = 0.0

            for term in terms:
                if term in fname_lower:
                    score += 5.0 * keyword_weights.get(term, 1.0)

            matches = pattern.findall(text_lower)
            for m in matches:
                m_low = m.lower()
                score += 1.0 * keyword_weights.get(m_low, 1.0)

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Lọc đa dạng hóa: Mỗi file lấy tối đa `max_per_doc` chunks
        doc_count = {}
        diverse_results = []
        for score, doc in scored:
            fn = doc["filename"]
            count = doc_count.get(fn, 0)
            if count < max_per_doc:
                diverse_results.append(doc)
                doc_count[fn] = count + 1
            if len(diverse_results) >= top_k:
                break

        return diverse_results
