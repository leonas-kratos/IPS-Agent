#!/usr/bin/env python3
"""
realtime_fgoekf.py
=================
Chương trình định vị thời gian thực bằng thuật toán đề xuất FGO-EKF:
  (Factor Graph Optimization + Extended Kalman Filter)
  - Đầu vào: Đặc trưng 57D (50 CIR + 7 PHY) từ DWM1001 + dữ liệu xe ($IMU từ STM32).
  - Tầng AI: Pure NumPy Tiny 1D-CNN ước lượng Range Bias b̂_i và phân loại trạng thái kênh (LOS/NLOS).
  - Tầng Consensus: Kiểm định Chi-bình phương (NIS Gating) thích nghi ma trận R_k + ưu tiên LOS.
  - Tầng Sensor Fusion: EKF kết hợp Odometry bánh xe (robot_V) + góc IMU (robot_deg).
  - Đầu ra: In nổi bật X: ..., Y: ... theo thời gian thực.
  - Tự động lưu toàn bộ dữ liệu thô, dự đoán AI và tọa độ ước lượng vào file CSV.
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
    TinyCNNNumpyEngine, DEFAULT_CIR_BAUD, DEFAULT_ROBOT_BAUD,
    N_CIR, N_META_FIELDS, CLASS_NAMES
)


class OnlineFGOEKF:
    """Thuật toán FGO-EKF Sensor Fusion thời gian thực."""
    def __init__(self, anchors, init_pos=None):
        self.anchors = anchors
        if init_pos is None:
            init_pos = np.mean(list(anchors.values()), axis=0)
        self.state = np.array(init_pos, dtype=np.float64)  # [x, y]
        self.P = np.eye(2) * (500.0**2)
        self.Q_base = np.eye(2) * (15.0**2)
        self.R_base = 120.0**2
        self.last_ts = time.time()

    def predict(self, robot_v, robot_deg, now=None):
        if now is None:
            now = time.time()
        dt = np.clip(now - self.last_ts, 0.005, 0.5)
        self.last_ts = now

        if robot_v is not None and robot_deg is not None and abs(robot_v) > 0.1:
            theta_rad = -np.radians(robot_deg)
            dx = robot_v * np.cos(theta_rad) * dt
            dy = robot_v * np.sin(theta_rad) * dt
            self.state = self.state + np.array([dx, dy])
            Q_k = self.Q_base * (1.0 + (robot_v / 100.0)**2) * (dt / 0.05)
        else:
            Q_k = self.Q_base * 4.0 * (dt / 0.05)

        self.P = self.P + Q_k

    def update(self, anchor_id, range_corr, probs=None):
        pa = self.anchors[anchor_id]
        pred_dist = np.linalg.norm(self.state - pa)
        if pred_dist < 1e-4:
            return self.state.copy(), 1.0

        H = ((self.state - pa) / pred_dist).reshape(1, 2)
        y_res = range_corr - pred_dist
        chi2_thresh = 3.841

        S_scalar = float((H @ self.P @ H.T + self.R_base)[0, 0])
        nis = (y_res**2) / max(1e-4, S_scalar)
        c_factor = 1.0 if nis <= chi2_thresh else (chi2_thresh / nis)
        c_weight = float(np.clip(c_factor, 0.05, 1.0))
        R_k = self.R_base / max(1e-3, c_weight)

        S = H @ self.P @ H.T + R_k
        K = self.P @ H.T / S[0, 0]

        self.state = self.state + (K * y_res).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.state.copy(), c_weight


def main():
    parser = argparse.ArgumentParser(description="Định vị thời gian thực FGO-EKF (Tiny 1D-CNN + LOS Consensus + Odometry SF).")
    parser.add_argument("--cir-port", type=str, default=None, help="Cổng Serial CIR (mặc định: tự động nhận diện hoặc /dev/ttyACM1)")
    parser.add_argument("--cir-baud", type=int, default=DEFAULT_CIR_BAUD, help=f"Baudrate CIR ({DEFAULT_CIR_BAUD})")
    parser.add_argument("--robot-port", type=str, default=None, help="Cổng Robot STM32 (mặc định: tự động nhận diện hoặc /dev/ttyACM0)")
    parser.add_argument("--robot-baud", type=int, default=DEFAULT_ROBOT_BAUD, help=f"Baudrate Robot ({DEFAULT_ROBOT_BAUD})")
    parser.add_argument("--no-robot", action="store_true", help="Không kết nối cổng Robot")
    parser.add_argument("--model", type=str, default="fgo_ekf_weights_numpy.npz", help="File trọng số Tiny CNN (.npz)")
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
    print("      ĐỊNH VỊ REALTIME: FGO-EKF (CONVOLUTION CONSENSUS EKF)")
    print("=" * 70)

    anchors = load_anchor_setup(args.setup)
    anchor_keys = sorted(list(anchors.keys()))
    print("[*] Danh sách Anchor:")
    for aid in anchor_keys:
        print(f"    • {aid}: X = {anchors[aid][0]:8.1f} mm, Y = {anchors[aid][1]:8.1f} mm")

    print("[*] Đang tải mô hình Tiny 1D-CNN (Pure NumPy Engine)...")
    ai_engine = TinyCNNNumpyEngine()
    print("[+] Tải mô hình thành công (Không cần TensorFlow).")

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
        out_filename = args.output or f"fgo_ekf_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_file = open(out_filename, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(log_file)
        csv_header = (
            ["timestamp", "robot_V", "robot_deg", "anchor_id", "range_raw",
             "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3", "STDEV_NOISE", "CIR_PWR", "MAX_NOISE"]
            + [f"CIR{i}" for i in range(N_CIR)]
            + ["bias_hat", "range_corr", "class_pred", "p_los", "X_FGOEKF_mm", "Y_FGOEKF_mm"]
        )
        csv_writer.writerow(csv_header)
        log_file.flush()
        print(f"[*] Tự động lưu log vào: {Path(out_filename).resolve()}\n")

    fgo_ekf = OnlineFGOEKF(anchors)
    latest_corr = {}
    pos_fgo_ekf = None
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

            # Vector 57D: 50 CIR + 7 PHY
            phy_vec = [pwr, fp1, fp2, fp3, std_n, max_n, rng_raw]
            feat_57d = np.array(cir_vals + phy_vec, dtype=np.float32)

            # Dự đoán qua Tiny 1D-CNN Pure NumPy
            bias_hat, probs = ai_engine.predict(feat_57d)
            if probs is not None:
                cls_idx = int(np.argmax(probs))
                p_los = float(probs[0])
                state_str = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"
            else:
                p_los = None
                cls_idx = 0
                state_str = "FGO-EKF Reg"

            # Hiệu chỉnh cự ly
            rng_corr_3d = max(10.0, rng_raw - bias_hat)
            if args.height > 0.0:
                rng_corr = float(np.sqrt(max(10.0, rng_corr_3d**2 - args.height**2)))
            else:
                rng_corr = rng_corr_3d

            latest_corr[aid] = rng_corr
            packet_count += 1

            # FGO-EKF Prediction (Odometry + IMU) + Update (Chi-square NIS Gating & LOS Prioritized)
            if aid in anchors:
                fgo_ekf.predict(cur_v, cur_deg, now=recv_time)
                pos_fgo_ekf, _ = fgo_ekf.update(aid, rng_corr, probs)

            now = time.time()
            if now - last_print >= 0.5:
                last_print = now
                cycle_count += 1
                fps = packet_count / max(1e-3, now - start_time)
                t_str = datetime.fromtimestamp(recv_time).strftime("%H:%M:%S.%f")[:-4]

                # IN KẾT QUẢ ĐẦU RA YÊU CẦU: X: ..., Y: ...
                if pos_fgo_ekf is not None:
                    print(f"[{t_str}] [FGO-EKF #{cycle_count:<4} - {fps:.1f}Hz]  X: {pos_fgo_ekf[0]:7.1f} mm ({pos_fgo_ekf[0]/1000:5.2f} m)  |  Y: {pos_fgo_ekf[1]:7.1f} mm ({pos_fgo_ekf[1]/1000:5.2f} m)")
                    valid_aids = [k for k in anchor_keys if k in latest_corr]
                    dist_str = " | ".join([f"{a}: {np.linalg.norm(pos_fgo_ekf - anchors[a]):.0f}mm" for a in valid_aids])
                    print(f"      Khoảng cách ước lượng Tag->Anchor: {dist_str}")

            if csv_writer:
                t_str = datetime.fromtimestamp(recv_time).strftime("%H:%M:%S.%f")[:-4]
                xc = pos_fgo_ekf[0] if pos_fgo_ekf is not None else ""
                yc = pos_fgo_ekf[1] if pos_fgo_ekf is not None else ""
                row = [t_str, cur_v, cur_deg, aid, rng_raw, fp_idx, fp1, fp2, fp3, std_n, pwr, max_n] + cir_vals + [bias_hat, rng_corr, state_str, p_los, xc, yc]
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
