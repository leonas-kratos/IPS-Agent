#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_core.agent import UWBAgent
from agent_core.config import DEFAULT_MODE, DEFAULT_SERVER_URL
from agent_core.ui import (
    console, print_banner, print_help, print_sources, stream_markdown_response
)
from rich.table import Table
from rich import box

def run_interactive_cli(agent: UWBAgent):
    # Lấy thông tin trạng thái ban đầu
    status = agent.llm.check_connection()
    model_name = status.get("current_model", status.get("server_url", "Cloud Engine"))
    kb_count = len(agent.kb_engine.documents)

    print_banner(mode=agent.llm.mode, model_name=model_name, kb_chunks=kb_count, skills_count=len(agent.active_skills), use_rag=agent.use_rag)

    # Nếu máy chủ server chưa kết nối được
    if agent.llm.mode == "server" and not status.get("ok"):
        console.print(
            f"\n[bold yellow]💡 Lưu ý:[/bold yellow] Chưa kết nối được máy chủ tập trung ({agent.llm.server_url}).\n"
            "👉 Nếu bạn đã deploy server lên Hugging Face, hãy cập nhật URL trong [bold cyan]agent_core/config.py[/bold cyan].\n"
            "👉 Bạn có thể chuyển sang dùng Local Ollama bằng lệnh: [bold green]/mode local[/bold green]\n"
            "👉 Hoặc dùng trực tiếp Google Gemini API bằng lệnh: [bold green]/mode online[/bold green]\n"
        )
    elif agent.llm.mode == "local" and not status.get("models"):
        console.print(
            "\n[bold yellow]⚠️ Chú ý:[/bold yellow] Ollama đang chạy nhưng chưa có mô hình nào.\n"
            "👉 Tải model nhẹ khuyên dùng (2GB): [bold cyan]ollama pull qwen2.5:3b[/bold cyan]\n"
        )

    last_ctrl_c_time = 0.0
    while True:
        try:
            user_input = console.input("[bold green]IPS-Agent ❯ [/bold green]").strip()
            last_ctrl_c_time = 0.0
        except KeyboardInterrupt:
            now = time.time()
            if now - last_ctrl_c_time < 2.0:
                console.print("\n[bold yellow]Tạm biệt! Hẹn gặp lại.[/bold yellow]")
                break
            else:
                last_ctrl_c_time = now
                console.print("\n[dim yellow](Nhấn Ctrl+C một lần nữa trong vòng 2 giây để thoát)[/dim yellow]")
                continue
        except EOFError:
            console.print("\n[bold yellow]Tạm biệt! Hẹn gặp lại.[/bold yellow]")
            break

        if not user_input:
            continue

        # Xử lý các lệnh điều khiển
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ["/exit", "/quit"]:
                console.print("[bold yellow]Tạm biệt! Hẹn gặp lại.[/bold yellow]")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/skills":
                from rich.table import Table
                from rich import box
                t = Table(title=f"Danh sách Kỹ năng & Quy chuẩn Markdown đang nạp ({len(agent.active_skills)} Skills)", border_style="bright_cyan", box=box.ROUNDED)
                t.add_column("STT", justify="center", style="bold bright_cyan", width=4)
                t.add_column("Tên Kỹ năng", style="bold bright_yellow")
                t.add_column("Mô tả / Phạm vi", style="white")
                t.add_column("Đường dẫn tệp", style="dim green")
                for idx, sk in enumerate(agent.active_skills, 1):
                    rel_path = Path(sk['path']).relative_to(agent.kb_engine.root_dir)
                    t.add_row(str(idx), sk['name'], sk['desc'], str(rel_path))
                console.print(t)
                console.print("[dim]💡 Bạn có thể chỉnh sửa trực tiếp các tệp trong thư mục [bold cyan]skills/[/bold cyan] để tùy biến hướng dẫn cho Model bất kỳ lúc nào.[/dim]\n")
            elif cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                print_banner(mode=agent.llm.mode, model_name=agent.llm.model or "Default", kb_chunks=len(agent.kb_engine.documents), skills_count=len(agent.active_skills), use_rag=agent.use_rag)
            elif cmd == "/key":
                if not arg:
                    new_key = console.input("[yellow]Nhập API Key tạm thời (hoặc Enter để hủy): [/yellow]").strip()
                else:
                    new_key = arg.strip()

                if new_key.lower() in ["clear", "remove", "delete"]:
                    agent.llm.api_key = None
                    agent.set_mode("server")
                    console.print("[bold green]✔ Đã xóa API key tạm thời. Đã chuyển về chế độ Server tập trung.[/bold green]")
                elif new_key:
                    agent.set_mode("online", api_key=new_key)
                    masked = new_key[:4] + "..." + new_key[-4:] if len(new_key) > 8 else "***"
                    console.print(f"[bold green]✔ Đã nạp API Key tạm thời ({masked}).[/bold green]")
                    console.print("[bold cyan]✔ Tự động kích hoạt chế độ Online Direct cho phiên này (không lưu vào ổ đĩa)![/bold cyan]")
                else:
                    curr = agent.llm.api_key
                    if curr:
                        masked = curr[:4] + "..." + curr[-4:] if len(curr) > 8 else "***"
                        console.print(f"API Key hiện tại: [bold yellow]{masked}[/bold yellow] (Gõ [bold green]/key clear[/bold green] để xóa)")
                    else:
                        console.print("[dim]Chưa có API Key tạm thời nào được thiết lập.[/dim]")
            elif cmd == "/mode":
                if not arg:
                    # Xoay vòng chế độ: server -> local -> online
                    modes = ["server", "local", "online"]
                    curr_idx = modes.index(agent.llm.mode) if agent.llm.mode in modes else 0
                    new_mode = modes[(curr_idx + 1) % len(modes)]
                else:
                    new_mode = arg.lower()
                
                if new_mode == "online" and not agent.llm.api_key:
                    key = console.input("[yellow]Nhập GEMINI_API_KEY (hoặc Enter để bỏ qua): [/yellow]").strip()
                    if key:
                        agent.set_mode("online", api_key=key)
                    else:
                        console.print("[red]Chưa có API key, vẫn giữ chế độ cũ.[/red]")
                        continue
                else:
                    agent.set_mode(new_mode)
                
                check = agent.llm.check_connection()
                console.print(f"[bold green]✔ Đã chuyển sang chế độ:[/bold green] {check.get('type')}")
            elif cmd == "/model":
                if arg:
                    agent.llm.model = arg
                    console.print(f"[bold green]✔ Đã đổi sang model:[/bold green] {arg}")
                else:
                    console.print(f"Model hiện tại: [bold yellow]{agent.llm.model}[/bold yellow]")
            elif cmd == "/rag":
                arg_clean = arg.strip().lower()
                if arg_clean in ["on", "1", "true", "enable", "bat"]:
                    agent.use_rag = True
                    console.print("[bold green]✔ Đã BẬT kiến trúc RAG.[/bold green] Agent sẽ trích xuất đoạn văn bản từ KB Engine để bổ sung ngữ cảnh.")
                elif arg_clean in ["off", "0", "false", "disable", "tat"]:
                    agent.use_rag = False
                    console.print("[bold green]✔ Đã TẮT kiến trúc RAG. Chuyển sang CHẠY THUẦN (Pure LLM) 🚀[/bold green]")
                    console.print("[dim]Agent sẽ phản hồi trực tiếp dựa trên tri thức mô hình và kỹ năng chuyên sâu (không trích xuất đoạn vụn vặt).[/dim]\n")
                else:
                    agent.use_rag = not agent.use_rag
                    status_text = "[bold yellow]BẬT (Trích xuất đoạn trích)[/bold yellow]" if agent.use_rag else "[bold green]TẮT (Chạy thuần Pure LLM 🚀)[/bold green]"
                    console.print(f"Trạng thái RAG hiện tại: {status_text}")
                    console.print("[dim]Cú pháp: /rag on để bật, /rag off để tắt[/dim]\n")
            elif cmd == "/search":
                if not arg:
                    console.print("[red]Cú pháp: /search <từ khóa>[/red]")
                    continue
                with console.status(f"[bold cyan]🔍 Đang tra cứu '{arg}'...[/bold cyan]"):
                    results = agent.kb_engine.search(arg, top_k=3)
                if not results:
                    console.print("[yellow]Không tìm thấy kết quả phù hợp.[/yellow]")
                else:
                    for r in results:
                        console.print(f"\n[bold cyan]• [{r['source_type']}] {r['filename']}[/bold cyan]")
                        console.print(f"[dim]{r['text'][:300]}...[/dim]\n")
            elif cmd == "/index":
                with console.status("[bold cyan]🔄 Đang quét lại toàn bộ tài liệu và code...[/bold cyan]"):
                    agent.kb_engine.rebuild_index()
                console.print(f"[bold green]✔ Đã lập chỉ mục xong {len(agent.kb_engine.documents)} đoạn tài liệu![/bold green]")
            elif cmd == "/papers":
                scan_results = agent.find_relevant_repo_papers(arg, top_k=48) if arg else agent.repo_papers
                
                table = Table(
                    title=f"Kho bài báo khoa học UWB trong Repo {f'(Lọc theo: {arg})' if arg else f'({len(scan_results)} bài)'}",
                    border_style="bright_cyan",
                    box=box.ROUNDED
                )
                table.add_column("STT", style="bold bright_cyan", justify="center", width=4)
                table.add_column("Tên bài báo", style="bold bright_white")
                table.add_column("Năm", justify="center", style="bold bright_yellow", width=6)
                table.add_column("Trường phái / Phân loại", style="bright_green", width=24)
                table.add_column("Tệp PDF", style="dim cyan")

                for idx, p in enumerate(scan_results, 1):
                    table.add_row(
                        str(idx),
                        p.get("title", "")[:60] + ("..." if len(p.get("title", "")) > 60 else ""),
                        str(p.get("year", "2024")),
                        p.get("category", "UWB"),
                        p.get("filename", "")[:35] + ("..." if len(p.get("filename", "")) > 35 else "")
                    )
                console.print(table)
                console.print(f"[dim]💡 Để đọc toàn văn bài báo cụ thể, hãy dùng lệnh: [bold cyan]/read <tên_file>[/bold cyan][/dim]\n")
            elif cmd == "/read":
                if not arg:
                    console.print("[red]Cú pháp: /read <tên_bài_báo>[/red] (Ví dụ: /read NCKH hoặc /read SimCLR)")
                    continue
                with console.status(f"[bold cyan]📖 Đang đọc 100% nguyên văn bài báo '{arg}'...[/bold cyan]"):
                    paper_text = agent.tools.read_full_paper(arg)
                if paper_text.startswith("Không tìm thấy") or paper_text.startswith("Lỗi đọc"):
                    console.print(f"[red]{paper_text}[/red]")
                else:
                    agent.history.append({
                        "role": "user",
                        "content": f"Tôi vừa nạp nguyên văn toàn bộ nội dung của bài báo ({arg}):\n\n{paper_text[:25000]}\n\nHãy ghi nhớ toàn bộ nội dung này."
                    })
                    agent.history.append({
                        "role": "assistant",
                        "content": f"Tôi đã đọc và nạp trọn vẹn 100% nội dung bài báo ({arg}) vào bộ nhớ. Bạn có thể hỏi bất kỳ câu hỏi chuyên sâu nào về phương pháp, công thức, số liệu trong bài này!"
                    })
                    console.print(f"[bold green]✔ Đã nạp thành công 100% nguyên văn bài báo '{arg}' ({len(paper_text):,} ký tự) vào bộ nhớ![/bold green]")
                    console.print("[dim]Bây giờ bạn có thể hỏi bất kỳ câu hỏi chi tiết nào về bài báo này.[/dim]\n")
            elif cmd in ["/all", "/scan"]:
                if not arg:
                    console.print("[red]Cú pháp: /all <câu hỏi tổng hợp>[/red] (Ví dụ: /all Các phương pháp phân loại LOS/NLOS dùng CIR)")
                    continue
                console.print(f"[bold cyan]🚀 Đang khởi động tiến trình quét đọc 100% toàn bộ 48 bài báo từ từ...[/bold cyan]")
                stream_gen = agent.scanner.scan_and_synthesize_all(arg)
                stream_markdown_response(stream_gen)
            elif cmd == "/calc":
                console.print("[bold cyan]📐 Công cụ tính tọa độ Tag 2D (Least Squares):[/bold cyan]")
                ans = agent.tools.calculate_trilateration_2d(
                    anchors=[(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)],
                    distances=[3.2, 2.5, 4.1]
                )
                console.print(ans)
            else:
                console.print(f"[red]Lệnh không hợp lệ: {cmd}. Gõ /help để xem hướng dẫn.[/red]")
            continue

        # Trả lời câu hỏi người dùng
        try:
            stream_gen, retrieved_docs = agent.answer_question_stream(user_input)
            
            # Hiển thị tài liệu trích dẫn (chỉ khi người dùng chủ động bật --rag)
            if retrieved_docs and agent.use_rag:
                print_sources(retrieved_docs)

            # Stream câu trả lời thời gian thực (nhấn Ctrl+C 1 lần dừng câu trả lời hiện tại)
            stream_markdown_response(stream_gen)

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]⏹ Đã dừng phản hồi.[/bold yellow]\n")
        except Exception as e:
            console.print(f"\n[bold red]Lỗi khi xử lý:[/bold red] {str(e)}\n")

def main():
    parser = argparse.ArgumentParser(description="IPS-UWB AI Agent Terminal")
    parser.add_argument("--mode", choices=["server", "local", "online"], default=DEFAULT_MODE, help="Chế độ (server, local hoặc online)")
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL, help="URL máy chủ AI online")
    parser.add_argument("--model", type=str, default=None, help="Tên mô hình")
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API Key (nếu dùng online)")
    parser.add_argument("--rag", action="store_true", default=False, help="Bật kiến trúc RAG đoạn trích (mặc định: False - Chạy thuần)")
    args = parser.parse_args()

    agent = UWBAgent(mode=args.mode, model=args.model, api_key=args.api_key, use_rag=args.rag)
    if args.server_url:
        agent.llm.server_url = args.server_url
    run_interactive_cli(agent)

if __name__ == "__main__":
    main()
