import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich import box

console = Console()

def print_banner(mode: str, model_name: str, kb_chunks: int, skills_count: int = 4, use_rag: bool = False):
    """Hiển thị Banner chào mừng mang phong cách Agent Terminal."""
    banner_text = Text()
    banner_text.append("📡 IPS-UWB AI AGENT TERMINAL\n", style="bold bright_cyan")
    banner_text.append("Trợ lý Nghiên cứu & Kỹ sư ảo chuyên sâu về Định vị UWB\n\n", style="italic white")
    banner_text.append("• Tri thức nạp sẵn: ", style="bold")
    banner_text.append(f"{kb_chunks} đoạn trích (54 Papers + Datasheet DWM1001 + Mã nguồn CAR/Tag)\n", style="bold green")
    banner_text.append("• Kỹ năng Agent (Skills): ", style="bold")
    banner_text.append(f"{skills_count} Skills Markdown (Master, NLOS-Mitigation, EKF-Fusion, DWM1001)\n", style="bold cyan")
    
    mode_color = "bright_magenta" if mode == "local" else "bright_blue"
    mode_label = "LOCAL (Ollama - 0 Limit, 0$)" if mode == "local" else "ONLINE (Groq / Gemini)"
    banner_text.append("• Chế độ hiện tại: ", style="bold")
    banner_text.append(f"[{mode_label}]", style=f"bold {mode_color}")
    banner_text.append("  | Model: ", style="bold")
    banner_text.append(f"{model_name}\n", style="bold yellow")
    
    banner_text.append("• Kiến trúc xử lý: ", style="bold")
    if use_rag:
        banner_text.append("[RAG - Tìm kiếm đoạn trích]\n\n", style="bold bright_yellow")
    else:
        banner_text.append("[CHẠY THUẦN (Pure LLM - Không cắt vụn dữ liệu) 🚀]\n\n", style="bold bright_green")

    banner_text.append("Gõ câu hỏi bất kỳ hoặc gõ ", style="dim")
    banner_text.append("/help", style="bold yellow")
    banner_text.append(" để xem các lệnh điều khiển.", style="dim")

    panel = Panel(
        banner_text,
        border_style="bright_blue",
        title="[bold white] Google Antigravity Style [/bold white]",
        subtitle="[dim]Gõ /exit để thoát[/dim]"
    )
    console.print(panel)

def print_help():
    """Hiển thị bảng trợ giúp lệnh kẻ ô bo tròn màu sắc."""
    table = Table(title="Danh sách lệnh điều khiển", border_style="bright_cyan", box=box.ROUNDED)
    table.add_column("Lệnh", style="bold yellow")
    table.add_column("Mô tả", style="white")
    table.add_row("/rag [on|off]", "Bật/Tắt kiến trúc RAG đoạn trích (Mặc định: TẮT - Chạy thuần Pure LLM)")
    table.add_row("/skills", "Xem danh sách các bộ kỹ năng & quy chuẩn (SKILL.md) nạp vào Model")
    table.add_row("/mode", "Chuyển đổi giữa chế độ Local (Ollama), Online (Groq) và Server")
    table.add_row("/key <api_key>", "Thêm API Key tạm thời cho phiên làm việc (hoặc /key clear để xóa)")
    table.add_row("/model <tên>", "Đổi model (VD: /model ips-uwb hoặc /model gemini-2.5-flash)")
    table.add_row("/papers [từ khóa]", "Tra cứu danh mục 48 bài báo UWB trong repo (Trường phái, Phương pháp, Tệp PDF)")
    table.add_row("/all <chủ đề>", "Quét đọc 100% toàn bộ 48 bài báo từ từ và viết báo cáo tổng hợp")
    table.add_row("/read <tên_file>", "Nạp nguyên văn 100% toàn bộ nội dung một bài báo vào bộ nhớ")
    table.add_row("/search <từ khóa>", "Tra cứu trực tiếp nội dung trong kho tài liệu không qua LLM")
    table.add_row("/calc", "Mở công cụ tính toán tọa độ Tag 2D bằng Least Squares")
    table.add_row("/index", "Quét lại và cập nhật cơ sở dữ liệu tri thức")
    table.add_row("/clear", "Xóa sạch màn hình terminal")
    table.add_row("/exit hoặc /quit", "Thoát chương trình")
    console.print(table)

def print_sources(sources):
    """Hiển thị các tài liệu đã tham khảo dưới dạng hộp bo góc có màu sắc."""
    if not sources:
        return
    text = Text()
    text.append("📚 Nguồn tài liệu tham khảo:\n", style="bold yellow")
    for s in sources:
        text.append(f"  • [{s.get('source_type')}] ", style="bold cyan")
        text.append(f"{s.get('filename')}\n", style="white")
    panel = Panel(text, border_style="bright_yellow", box=box.ROUNDED, title="[bold yellow]Knowledge Retrieval[/bold yellow]")
    console.print(panel)

def _colorize_inline_markdown(line: str) -> str:
    """Tô màu phong phú cho cú pháp markdown bên trong dòng văn bản (tránh chữ toàn màu trắng)."""
    # Inline code: `code`
    line = re.sub(r'`([^`]+)`', r'[bold bright_green]\1[/bold bright_green]', line)
    # Bold with colon: **Tiêu đề:** -> [bold bright_yellow]Tiêu đề:[/bold bright_yellow]
    line = re.sub(r'\*\*([^\*]+:?)\*\*', r'[bold bright_yellow]\1[/bold bright_yellow]', line)
    # Italic: *chữ nghiêng*
    line = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'[italic cyan]\1[/italic cyan]', line)
    # Tên bài báo trong ngoặc vuông hoặc ngoặc kép: "Tên bài" hoặc [Tên bài]
    line = re.sub(r'\[(\d+)\]', r'[bold bright_magenta][\1][/bold bright_magenta]', line)
    return line

def _render_styled_markdown(text: str):
    """
    Render văn bản Markdown chuyên nghiệp với:
    1. Bảng kẻ ô chuẩn chỉ bo góc ROUNDED, phân chia cột rõ ràng và tiêu đề màu sắc nổi bật.
    2. Chữ văn bản có màu sắc phong phú, không bị tình trạng 'toàn màu trắng'.
    3. Tiêu đề phân cấp trực quan với icon và màu sắc riêng biệt.
    """
    lines = text.split("\n")
    i = 0
    in_code_block = False
    code_lines = []

    while i < len(lines):
        line = lines[i]

        # Khối Code block (```)
        if line.strip().startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                console.print(Panel(code_text, border_style="bright_green", box=box.ROUNDED))
                code_lines = []
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Phát hiện Bảng Markdown (| ... |)
        if line.strip().startswith("|") and i + 1 < len(lines) and "---" in lines[i + 1]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            
            if len(table_lines) >= 3:
                # Trích xuất Header
                raw_headers = [re.sub(r"\*+", "", h).strip() for h in table_lines[0].split("|")[1:-1]]
                table = Table(
                    box=box.ROUNDED, 
                    border_style="bright_cyan", 
                    header_style="bold bright_yellow",
                    row_styles=["none", "dim"]
                )
                for h in raw_headers:
                    # Tự động canh lề cho cột STT hoặc Năm
                    if h.lower() in ["stt", "#", "năm", "year"]:
                        table.add_column(h, justify="center", style="bold bright_cyan")
                    elif h.lower() in ["loại", "trường phái", "nhóm", "phân loại"]:
                        table.add_column(h, style="bold bright_green")
                    elif "bài báo" in h.lower() or "tên" in h.lower() or "title" in h.lower():
                        table.add_column(h, style="bold bright_white")
                    else:
                        table.add_column(h, style="white")

                # Trích xuất Rows
                for row_line in table_lines[2:]:
                    cells = [c.strip() for c in row_line.split("|")[1:-1]]
                    if not any(c for c in cells):
                        continue
                    if len(cells) < len(raw_headers):
                        cells += [""] * (len(raw_headers) - len(cells))
                    elif len(cells) > len(raw_headers):
                        cells = cells[:len(raw_headers)]
                    
                    # Tô màu các cell
                    styled_cells = [_colorize_inline_markdown(c) for c in cells]
                    table.add_row(*styled_cells)

                console.print(table)
            continue

        # Tiêu đề Markdown với màu sắc trực quan sinh động
        if line.startswith("# "):
            console.print(f"\n[bold bright_cyan]📌 {line[2:].strip()}[/bold bright_cyan]")
        elif line.startswith("## "):
            console.print(f"\n[bold bright_green]🔹 {line[3:].strip()}[/bold bright_green]")
        elif line.startswith("### "):
            console.print(f"\n[bold bright_yellow]🔸 {line[4:].strip()}[/bold bright_yellow]")
        elif line.startswith("#### "):
            console.print(f"\n[bold yellow]▪ {line[5:].strip()}[/bold yellow]")
        elif line.startswith("> "):
            console.print(f"[italic bright_cyan]│ {_colorize_inline_markdown(line[2:].strip())}[/italic bright_cyan]")
        elif line.strip().startswith(("- ", "* ")):
            content = line.strip()[2:].strip()
            console.print(f"  [bright_green]•[/bright_green] {_colorize_inline_markdown(content)}")
        elif re.match(r"^\d+\.\s", line.strip()):
            m = re.match(r"^(\d+\.)\s(.*)", line.strip())
            num_str, rest = m.group(1), m.group(2)
            console.print(f"  [bold bright_cyan]{num_str}[/bold bright_cyan] {_colorize_inline_markdown(rest)}")
        elif line.strip() in ["---", "***", "___"]:
            console.print("[dim cyan]" + "─" * 70 + "[/dim cyan]")
        else:
            if line.strip():
                # Văn bản thông thường được tô màu sáng rõ ràng
                console.print(_colorize_inline_markdown(line))
            else:
                print()
        i += 1

import sys

def stream_markdown_response(generator):
    """
    Hiển thị câu trả lời thời gian thực (Realtime Streaming) từng token/chunk tới terminal,
    không gom buffer chờ đợi. Hỗ trợ ngắt dừng tức thì khi nhấn Ctrl+C 1 lần.
    """
    full_text = ""
    print()
    try:
        for chunk in generator:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_text += chunk
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        console.print("[bold yellow]⏹ Đã dừng phản hồi.[/bold yellow]\n")
        return full_text

    print("\n")
    return full_text

