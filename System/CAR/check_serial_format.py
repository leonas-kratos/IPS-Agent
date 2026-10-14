#!/usr/bin/env python3
"""
check_serial_format.py
======================
Công cụ kiểm tra và soi định dạng dữ liệu thô từ các cổng Serial (STM32 Robot và DWM1001 CIR).

Tính năng:
1. Tự động quét tất cả cổng /dev/ttyACM* và /dev/ttyUSB*.
2. Nhận diện cổng Robot (phát $IMU) và cổng CIR (phát 0x... ở 921600 baud).
3. Phân tích chi tiết từng trường dữ liệu:
   - Cổng Robot: in các dòng $IMU hoặc log khởi động STM32.
   - Cổng CIR: đếm số trường tách bằng dấu phẩy (yêu cầu 59 trường = 9 metadata + 50 CIR).
4. Chế độ Live Monitor: xem song song dữ liệu từ cả 2 cổng theo thời gian thực.
"""

import argparse
import glob
import sys
import time
import serial
import serial.tools.list_ports


def scan_available_ports():
    """Tìm tất cả các cổng /dev/ttyACM* và /dev/ttyUSB* khả dụng."""
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        ports = sorted([p.device for p in serial.tools.list_ports.comports() if not p.device.startswith("/dev/ttyS")])
    return ports


def read_lines_from_port(port, baudrate, duration=1.5, max_lines=15):
    """Đọc an toàn các dòng dữ liệu từ cổng serial với timeout ngắn, không bị treo."""
    lines = []
    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as ser:
            ser.dtr = True
            ser.rts = True
            time.sleep(0.15)
            ser.reset_input_buffer()
            start = time.time()
            buf = ""
            while time.time() - start < duration and len(lines) < max_lines:
                chunk = ser.read(ser.in_waiting or 1).decode("utf-8", errors="replace")
                if chunk:
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            lines.append(line)
                else:
                    time.sleep(0.02)
    except Exception as e:
        print(f"[-] Lỗi khi mở {port} ở baud {baudrate}: {e}")
    return lines


def inspect_single_port(port):
    """Kiểm tra và phân tích một cổng serial."""
    print("=" * 70)
    print(f"[*] KIỂM TRA CỔNG: {port}")
    print("=" * 70)

    # Ưu tiên thử 921600 baud trước vì CIR chỉ chạy ở 921600, còn STM32 USB CDC chạy được ở cả 2
    print(f"[*] Đang đọc thử ở Baudrate 921600 (chờ ~1.5s)...")
    lines_921 = read_lines_from_port(port, 921600, duration=1.5)

    is_robot = False
    is_cir = False

    if lines_921:
        imu_in_921 = sum(1 for l in lines_921 if "$IMU" in l)
        cir_in_921 = sum(1 for l in lines_921 if l.lower().startswith("0x") or ("," in l and len(l.split(",")) > 10))

        if imu_in_921 > 0:
            is_robot = True
            print(f"[+] Nhận diện: Đây là CỔNG XE ROBOT (STM32)!")
            print(f"    - Tìm thấy {imu_in_921} dòng $IMU.")
            print(f"    - Mẫu dòng: {lines_921[0]}")
        elif cir_in_921 > 0:
            is_cir = True
            print(f"[+] Nhận diện: Đây là CỔNG CIR (DWM1001)!")
            print(f"    - Tìm thấy {cir_in_921} dòng dữ liệu CIR.")
            analyze_cir_lines(lines_921)
        else:
            print(f"[?] Nhận được {len(lines_921)} dòng dữ liệu khác:")
            for l in lines_921[:3]:
                print(f"    {l[:100]}")
    else:
        print(f"[-] Không nhận được dữ liệu ở 921600 baud.")

    # Nếu chưa nhận diện được hoặc là Robot, thử ở 115200 baud
    if not is_cir:
        print(f"\n[*] Đang đọc thử ở Baudrate 115200 (chờ ~1.5s)...")
        lines_115 = read_lines_from_port(port, 115200, duration=1.5)
        if lines_115:
            imu_in_115 = sum(1 for l in lines_115 if "$IMU" in l)
            log_in_115 = sum(1 for l in lines_115 if any(k in l for k in ["STM32", "Robot", "BNO", "NAV", "Zero raw"]))
            if imu_in_115 > 0 or log_in_115 > 0:
                is_robot = True
                print(f"[+] Nhận diện: CỔNG XE ROBOT (STM32) ở 115200 baud!")
                print(f"    - Dòng $IMU: {imu_in_115}, Dòng log khởi động: {log_in_115}")
                for idx, l in enumerate(lines_115[:3], 1):
                    print(f"    Dòng {idx}: {l}")
        else:
            print(f"[-] Không nhận được dữ liệu ở 115200 baud.")

    role = "ROBOT" if is_robot else ("CIR" if is_cir else "UNKNOWN")
    print(f"--> KẾT LUẬN CỔNG {port}: [{role}]\n")
    return role


def analyze_cir_lines(lines):
    """Phân tích chi tiết các trường của dòng CIR."""
    meta_names = ["anchor_id", "range", "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3", "STDEV_NOISE", "CIR_PWR", "MAX_NOISE"]
    cir_lines = [l for l in lines if l.lower().startswith("0x") or ("," in l and len(l.split(",")) > 10)]

    if not cir_lines:
        return

    sample_line = cir_lines[0].rstrip("|").strip()
    parts = [p.strip() for p in sample_line.split(",") if p.strip()]
    total_fields = len(parts)

    print(f"\n    --- PHÂN TÍCH CHI TIẾT ĐỊNH DẠNG CIR ---")
    print(f"    • Tổng số trường nhận được : {total_fields} trường")
    print(f"    • Số trường yêu cầu chuẩn   : 59 trường (9 metadata + 50 giá trị CIR)")

    if total_fields == 59:
        print(f"    ===> ĐỊNH DẠNG HOÀN TOÀN KHỚP VỚI CODE PYTHON (59/59)!\n")
    else:
        print(f"    [!] CẢNH BÁO: Số trường hiện tại là {total_fields} (khác 59).")
        print(f"        Nếu số mẫu CIR = {total_fields - 9}, hãy chạy script với: --n-cir {total_fields - 9}\n")

    print(f"    • Phân tích 9 trường đầu tiên:")
    for idx, (m_name, val) in enumerate(zip(meta_names, parts[:9])):
        print(f"        [{idx}] {m_name:<12} = {val}")

    cir_samples = parts[9:]
    print(f"    • Số mẫu CIR thực tế: {len(cir_samples)} mẫu")
    print(f"        Mẫu 5 giá trị CIR đầu tiên : {cir_samples[:5]}")
    print(f"        Mẫu 5 giá trị CIR cuối cùng: {cir_samples[-5:]}")


def live_monitor(cir_port, robot_port):
    """Hiển thị dữ liệu song song từ 2 cổng theo thời gian thực."""
    print("=" * 70)
    print("        CHẾ ĐỘ THEO DÕI TRỰC TIẾP (LIVE MONITOR)")
    if cir_port:
        print(f"[*] Cổng CIR   : {cir_port} @ 921600 baud")
    if robot_port:
        print(f"[*] Cổng Robot : {robot_port} @ 115200 baud")
    print("Nhấn Ctrl+C để thoát.")
    print("=" * 70 + "\n")

    ser_cir = None
    ser_robot = None

    try:
        if cir_port:
            ser_cir = serial.Serial(cir_port, baudrate=921600, timeout=0.1)
            ser_cir.dtr = True
            ser_cir.rts = True
            ser_cir.reset_input_buffer()
        if robot_port:
            ser_robot = serial.Serial(robot_port, baudrate=115200, timeout=0.1)
            ser_robot.dtr = True
            ser_robot.rts = True
            ser_robot.reset_input_buffer()

        while True:
            # Đọc từ Robot
            if ser_robot and ser_robot.in_waiting:
                try:
                    r_line = ser_robot.readline().decode("utf-8", errors="replace").strip()
                    if r_line:
                        print(f"[ROBOT | {robot_port}]: {r_line}")
                except Exception:
                    pass

            # Đọc từ CIR
            if ser_cir and ser_cir.in_waiting:
                try:
                    c_line = ser_cir.readline().decode("utf-8", errors="replace").strip()
                    if c_line:
                        clean = c_line.rstrip("|").strip()
                        parts = [p.strip() for p in clean.split(",") if p.strip()]
                        anchor = parts[0] if parts else ""
                        print(f"[CIR   | {cir_port}]: {anchor} ({len(parts)} trường) | {c_line[:80]}...")
                except Exception:
                    pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[!] Dừng Live Monitor.")
    finally:
        if ser_cir:
            ser_cir.close()
        if ser_robot:
            ser_robot.close()


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra và soi định dạng dữ liệu từ STM32 Robot và DWM1001 CIR.")
    parser.add_argument("-p", "--port", type=str, default=None, help="Kiểm tra một cổng cụ thể (VD: /dev/ttyACM0)")
    parser.add_argument("--live", action="store_true", help="Chạy chế độ Live Monitor cả 2 cổng")
    args = parser.parse_args()

    available_ports = scan_available_ports()
    print("=" * 70)
    print("   KIỂM TRA CỔNG SERIAL VÀ ĐỊNH DẠNG DỮ LIỆU ROBOT / CIR")
    print("=" * 70)
    print(f"[*] Các cổng tìm thấy trên hệ thống: {available_ports if available_ports else 'KHÔNG CÓ'}")

    if not available_ports and not args.port:
        print("[-] Không tìm thấy cổng /dev/ttyACM* hoặc /dev/ttyUSB* nào!")
        print("    Vui lòng cắm cáp kết nối STM32 và DWM1001 vào máy.")
        sys.exit(1)

    ports_to_check = [args.port] if args.port else available_ports

    detected_robot = None
    detected_cir = None

    for port in ports_to_check:
        role = inspect_single_port(port)
        if role == "ROBOT":
            detected_robot = port
        elif role == "CIR":
            detected_cir = port

    # Nếu có 2 cổng và phát hiện được 1 cổng, cổng kia là cổng còn lại
    if len(ports_to_check) == 2:
        if detected_robot and not detected_cir:
            detected_cir = next(p for p in ports_to_check if p != detected_robot)
        elif detected_cir and not detected_robot:
            detected_robot = next(p for p in ports_to_check if p != detected_cir)

    print("=" * 70)
    print("                       TỔNG KẾT NHẬN DIỆN")
    print("=" * 70)
    print(f"  • Cổng Xe Robot (STM32, 115200 baud) : {detected_robot or 'Chưa xác định'}")
    print(f"  • Cổng CIR      (DWM1001, 921600 baud): {detected_cir or 'Chưa xác định'}")
    print("=" * 70 + "\n")

    if detected_robot and detected_cir:
        print(f"[+] Để chạy thu thập dữ liệu với đúng cổng đã nhận diện:")
        print(f"    python3 collect_data_trajec.py --robot-port {detected_robot} --cir-port {detected_cir}\n")

    if args.live or (detected_robot and detected_cir):
        ans = input(">> Bạn có muốn vào chế độ Live Monitor xem dữ liệu trực tiếp không? [Y/n]: ").strip().lower()
        if ans != 'n':
            live_monitor(detected_cir, detected_robot)


if __name__ == "__main__":
    main()
