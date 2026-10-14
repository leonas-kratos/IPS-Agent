#!/usr/bin/env python3
"""
realtime_lse.py
===============
Chương trình định vị thời gian thực bằng thuật toán LSE (Least Squares Estimation / Multilateration):
  - Đầu vào: Cự ly UWB thô từ cổng Serial DWM1001 (/dev/ttyACM1) + dữ liệu xe ($IMU từ /dev/ttyACM0).
  - Thuật toán: Multilateration hình học phi tuyến (hoặc tuyến tính).
  - Đầu ra: In nổi bật X: ..., Y: ... theo thời gian thực.
  - Tự động lưu toàn bộ dữ liệu thô và tọa độ ước lượng vào file CSV.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
import signal
import sys
import threading
import time

import numpy as np

from uwb_common import (
    load_anchor_setup, auto_detect_ports,
    RobotReader, CirReader, SimReaderThread, KeyboardListener,
    solve_lse, DEFAULT_CIR_BAUD, DEFAULT_ROBOT_BAUD,
    N_CIR, N_META_FIELDS
)


def main():
    parser = argparse.ArgumentParser(description="Định vị thời gian thực LSE (Multilateration thuần).")
    parser.add_argument("--cir-port", type=str, default=None, help="Cổng Serial CIR (mặc định: tự động nhận diện hoặc /dev/ttyACM1)")
    parser.add_argument("--cir-baud", type=int, default=DEFAULT_CIR_BAUD, help=f"Baudrate CIR ({DEFAULT_CIR_BAUD})")
    parser.add_argument("--robot-port", type=str, default=None, help="Cổng Robot STM32 (mặc định: tự động nhận diện hoặc /dev/ttyACM0)")
    parser.add_argument("--robot-baud", type=int, default=DEFAULT_ROBOT_BAUD, help=f"Baudrate Robot ({DEFAULT_ROBOT_BAUD})")
    parser.add_argument("--no-robot", action="store_true", help="Không kết nối cổng Robot")
    parser.add_argument("--setup", type=str, default="setup.txt", help="File tọa độ Anchor (setup.txt)")
    parser.add_argument("--height", type=float, default=0.0, help="Chênh lệch độ cao Anchor và Tag H (mm)")
    parser.add_argument("--no-save", action="store_true", help="Tắt tự động lưu CSV")
    parser.add_argument("-o", "--output", type=str, default=None, help="Tên file CSV lưu kết quả")
    parser.add_argument("--sim", type=str, default=None, help="File CSV để replay giả lập")
    parser.add_argument("--sim-fps", type=float, default=30.0, help="Tốc độ replay giả lập (Hz)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    print("=" * 70)
    print("      ĐỊNH VỊ REALTIME: LSE (LEAST SQUARES ESTIMATION)")
    print("=" * 70)

    anchors = load_anchor_setup(args.setup)
    anchor_keys = sorted(list(anchors.keys()))
    print("[*] Danh sách Anchor:")
    for aid in anchor_keys:
        print(f"    • {aid}: X = {anchors[aid][0]:8.1f} mm, Y = {anchors[aid][1]:8.1f} mm")

    stop_event = threading.Event()
    cir_queue = Queue()

    sim_reader = None
    robot_reader = None
    cir_reader = None

    if args.sim:
        print(f"[*] Chế độ GIẢ LẬP: Replay từ {args.sim} ({args.sim_fps} Hz)...")
        sim_reader = SimReaderThread(args.sim, args.sim_fps, cir_queue, stop_event)
        sim_reader.start()
        time.sleep(0.3)
    else:
        if not args.no_robot:
            cir_port, robot_port = auto_detect_ports(args.cir_port, args.robot_port)
        else:
            cir_port = args.cir_port or "/dev/ttyACM1"
            robot_port = None

        cir_reader = CirReader(cir_port, args.cir_baud, cir_queue, stop_event)
        cir_reader.start()
        time.sleep(0.3)
        if not cir_reader.connected:
            print(f"[-] Lỗi cổng CIR {cir_port}: {cir_reader.last_error}")
            sys.exit(1)
        print(f"[+] Kết nối cổng CIR: {cir_port}")

        if not args.no_robot:
            robot_reader = RobotReader(robot_port, args.robot_baud, stop_event)
            robot_reader.start()
            time.sleep(0.3)
            if robot_reader.connected:
                print(f"[+] Kết nối cổng Robot: {robot_port}")
            else:
                robot_reader = None

    kb_listener = None
    if robot_reader:
        kb_listener = KeyboardListener(robot_reader, stop_event)
        robot_reader.kb_listener = kb_listener
        kb_listener.start()

    # Mặc định tự động lưu CSV
    csv_writer = None
    log_file = None
    if not args.no_save:
        out_filename = args.output or f"lse_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_file = open(out_filename, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(log_file)
        csv_header = (
            ["timestamp", "robot_V", "robot_deg", "anchor_id", "range",
             "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3", "STDEV_NOISE", "CIR_PWR", "MAX_NOISE"]
            + [f"CIR{i}" for i in range(N_CIR)]
            + ["X_LSE_mm", "Y_LSE_mm"]
        )
        csv_writer.writerow(csv_header)
        log_file.flush()
        print(f"[*] Tự động lưu log vào: {Path(out_filename).resolve()}\n")

    latest_ranges = {}
    pos_lse = None
    packet_count = 0
    cycle_count = 0
    start_time = time.time()
    last_print = start_time

    print("-" * 70)
    try:
        while not stop_event.is_set():
            if robot_reader and robot_reader.check_done():
                print("\n[+] TÍN HIỆU TỪ XE: Robot đã hoàn thành lộ trình ($ROBOT,DONE)!")
                break

            try:
                recv_time, raw_line = cir_queue.get(timeout=0.1)
            except Empty:
                continue

            parts = [p.strip() for p in raw_line.rstrip("|").split(",") if p.strip()]
            if len(parts) < N_META_FIELDS + N_CIR:
                continue

            aid = parts[0].strip().lower()
            if not aid.startswith("0x"):
                continue

            try:
                rng_raw = float(parts[1])
                fp_idx = float(parts[2])
                fp1, fp2, fp3 = float(parts[3]), float(parts[4]), float(parts[5])
                std_n, pwr, max_n = float(parts[6]), float(parts[7]), float(parts[8])
                cir_vals = []
                for x in parts[N_META_FIELDS:N_META_FIELDS + N_CIR]:
                    try:
                        cir_vals.append(float(int(x, 16)))
                    except ValueError:
                        cir_vals.append(float(x))
            except ValueError:
                continue

            cur_v, cur_deg = 0.0, 0.0
            if robot_reader:
                cur_v, cur_deg, _ = robot_reader.get_latest()
            elif sim_reader:
                cur_v, cur_deg, _ = sim_reader.get_latest()

            # Hiệu chỉnh độ cao 3D -> 2D
            if args.height > 0.0:
                rng_2d = float(np.sqrt(max(10.0, rng_raw**2 - args.height**2)))
            else:
                rng_2d = rng_raw

            latest_ranges[aid] = rng_2d
            packet_count += 1

            # Khi có >= 3 anchors, giải LSE
            valid_aids = [k for k in anchor_keys if k in latest_ranges]
            if len(valid_aids) >= 3:
                pos_lse = solve_lse(valid_aids, anchors, latest_ranges, init_pos=pos_lse)

                now = time.time()
                if now - last_print >= 0.5 or len(valid_aids) == len(anchor_keys):
                    last_print = now
                    cycle_count += 1
                    fps = packet_count / max(1e-3, now - start_time)
                    t_str = datetime.fromtimestamp(recv_time).strftime("%H:%M:%S.%f")[:-4]

                    # IN KẾT QUẢ ĐẦU RA YÊU CẦU: X: ..., Y: ...
                    print(f"[{t_str}] [LSE #{cycle_count:<4} - {fps:.1f}Hz]  X: {pos_lse[0]:7.1f} mm ({pos_lse[0]/1000:5.2f} m)  |  Y: {pos_lse[1]:7.1f} mm ({pos_lse[1]/1000:5.2f} m)")
                    dist_str = " | ".join([f"{a}: {latest_ranges[a]:.0f}mm" for a in valid_aids])
                    print(f"      Khoảng cách đo: {dist_str}")

            if csv_writer:
                t_str = datetime.fromtimestamp(recv_time).strftime("%H:%M:%S.%f")[:-4]
                xl = pos_lse[0] if pos_lse is not None else ""
                yl = pos_lse[1] if pos_lse is not None else ""
                row = [t_str, cur_v, cur_deg, aid, rng_raw, fp_idx, fp1, fp2, fp3, std_n, pwr, max_n] + cir_vals + [xl, yl]
                csv_writer.writerow(row)
                if packet_count % 10 == 0:
                    log_file.flush()

    except (KeyboardInterrupt, SystemExit):
        print("\n[!] Dừng khẩn cấp.")
    finally:
        stop_event.set()
        if robot_reader:
            robot_reader.send_stop()
            robot_reader.join(timeout=1.0)
        if cir_reader:
            cir_reader.join(timeout=1.0)
        if sim_reader:
            sim_reader.join(timeout=1.0)
        if kb_listener:
            kb_listener.cleanup()
        if log_file:
            log_file.flush()
            log_file.close()
            print(f"[+] Đã lưu log thành công vào file: {out_filename}")


if __name__ == "__main__":
    main()
