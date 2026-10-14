#!/usr/bin/env python3
"""
uwb_common.py
=============
Module dùng chung cho các chương trình định vị thời gian thực trên Raspberry Pi 4:
  - Tự động nhận diện cổng Serial (Robot STM32 $IMU & DWM1001 CIR 59 trường)
  - Các luồng đọc đa luồng (RobotReader, CirReader, SimReaderThread)
  - Điều khiển xe robot qua bàn phím non-blocking (KeyboardListener)
  - Bộ suy luận Pure NumPy Tiny 1D-CNN (57D -> bias_hat, p_los) không cần TensorFlow
  - Giải thuật Multilateration LSE (Linear & Non-linear)
"""

import glob
import os
from pathlib import Path
from queue import Queue, Empty
import re
import select
import sys
import threading
import time

import numpy as np

try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import serial
import serial.tools.list_ports

DEFAULT_CIR_PORT = "/dev/ttyACM1"
DEFAULT_CIR_BAUD = 921600
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_BAUD = 115200

N_CIR = 50
N_META_FIELDS = 9
N_IN = 57
CLASS_NAMES = ["LOS", "NLOS (Board)", "NLOS (Table)"]


def resolve_file(filename: str) -> Path:
    p = Path(filename)
    if p.exists():
        return p
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "models" / filename,
        Path(__file__).resolve().parent.parent / filename,
        Path(__file__).resolve().parent.parent / "models" / filename,
        Path("models") / filename,
        Path("forPi/models") / filename,
        Path("Data/Moving") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def load_anchor_setup(setup_path="setup.txt"):
    anchors = {
        "0x1001": np.array([0.0, 0.0]),
        "0x1002": np.array([0.0, 5909.0]),
        "0x1003": np.array([10236.0, 5909.0]),
        "0x1004": np.array([10236.0, 0.0]),
    }
    actual_path = resolve_file(str(setup_path))
    if actual_path.exists():
        with open(actual_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"(0x[0-9a-fA-F]+)\s*:\s*([-\d.]+)\s*,\s*([-\d.]+)", line)
                if m:
                    aid = m.group(1).lower()
                    anchors[aid] = np.array([float(m.group(2)), float(m.group(3))])
    return anchors


def auto_detect_ports(user_cir_port=None, user_robot_port=None):
    if user_cir_port is not None and user_robot_port is not None:
        return user_cir_port, user_robot_port

    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not candidates:
        try:
            candidates = sorted([p.device for p in serial.tools.list_ports.comports() if not p.device.startswith("/dev/ttyS")])
        except Exception:
            candidates = []

    print("[*] Đang quét và nhận diện cổng serial...")
    detected_robot = user_robot_port
    detected_cir = user_cir_port

    if detected_robot is None:
        for port in candidates:
            if port == detected_cir:
                continue
            try:
                with serial.Serial(port, baudrate=DEFAULT_ROBOT_BAUD, timeout=0.2) as ser:
                    ser.dtr = True
                    ser.rts = True
                    time.sleep(0.1)
                    ser.reset_input_buffer()
                    start = time.time()
                    while time.time() - start < 0.8:
                        line = ser.readline().decode("utf-8", errors="ignore")
                        if any(k in line for k in ["$IMU", "STM32", "Robot", "BNO", "NAV", "Zero raw"]):
                            detected_robot = port
                            break
            except Exception:
                continue
            if detected_robot is not None:
                break

    if detected_robot is not None and detected_cir is None:
        remaining = [p for p in candidates if p != detected_robot]
        if remaining:
            detected_cir = remaining[0]

    if detected_cir is None:
        for port in candidates:
            if port == detected_robot:
                continue
            try:
                with serial.Serial(port, baudrate=DEFAULT_CIR_BAUD, timeout=0.2) as ser:
                    ser.dtr = True
                    ser.rts = True
                    time.sleep(0.1)
                    ser.reset_input_buffer()
                    start = time.time()
                    while time.time() - start < 0.8:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if line.startswith("0x") or line.startswith("0X") or ("," in line and len(line.split(",")) > 10):
                            detected_cir = port
                            break
            except Exception:
                continue
            if detected_cir is not None:
                break
        if detected_cir is not None and detected_robot is None:
            remaining = [p for p in candidates if p != detected_cir]
            if remaining:
                detected_robot = remaining[0]

    if detected_robot is None and detected_cir is None:
        if len(candidates) >= 2:
            detected_robot = candidates[0]
            detected_cir = candidates[1]
        elif len(candidates) == 1:
            detected_robot = candidates[0]
            detected_cir = DEFAULT_CIR_PORT
        else:
            detected_robot = DEFAULT_ROBOT_PORT
            detected_cir = DEFAULT_CIR_PORT
    elif detected_robot is None:
        remaining = [p for p in candidates if p != detected_cir]
        detected_robot = remaining[0] if remaining else DEFAULT_ROBOT_PORT
    elif detected_cir is None:
        remaining = [p for p in candidates if p != detected_robot]
        detected_cir = remaining[0] if remaining else DEFAULT_CIR_PORT

    print(f"[+] Nhận diện cổng: Robot={detected_robot} @ 115200 | CIR={detected_cir} @ 921600")
    return detected_cir, detected_robot


class RobotReader(threading.Thread):
    def __init__(self, port, baud, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self._latest_v = 0.0
        self._latest_deg = 0.0
        self._latest_ts = None
        self.count = 0
        self.ser = None
        self.connected = False
        self.last_error = None
        self.is_done = False
        self.kb_listener = None

    def get_latest(self):
        with self.lock:
            return self._latest_v, self._latest_deg, self._latest_ts

    def check_done(self):
        with self.lock:
            return self.is_done

    def send_start(self, kp=None):
        if not self.ser or not self.connected:
            return False
        try:
            if kp is not None:
                self.ser.write(f"CMD,KP,{kp}\n".encode("utf-8"))
                self.ser.flush()
                time.sleep(0.1)
            self.ser.write(b"CMD,START\n")
            self.ser.flush()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def send_stop(self):
        if not self.ser or not self.connected:
            return False
        try:
            self.ser.write(b"CMD,STOP\n")
            self.ser.flush()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def send_resume(self):
        if not self.ser or not self.connected:
            return False
        try:
            self.ser.write(b"CMD,RESUME\n")
            self.ser.flush()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def send_turn(self):
        if not self.ser or not self.connected:
            return False
        try:
            self.ser.write(b"CMD,TURN\n")
            self.ser.flush()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def send_trim(self, trim_l, trim_r):
        if not self.ser or not self.connected:
            return False
        try:
            cmd = f"CMD,TRIM,{int(trim_l)},{int(trim_r)}\n"
            self.ser.write(cmd.encode("utf-8"))
            self.ser.flush()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def run(self):
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=1.0)
            self.ser.dtr = True
            self.ser.rts = True
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.connected = True
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            return

        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception as e:
                self.last_error = str(e)
                continue
            if not raw:
                continue

            if "$ROBOT,DONE" in raw or "$DONE" in raw:
                with self.lock:
                    self.is_done = True
                continue

            if "[NAV]" in raw and ("Canh" in raw or "Quay" in raw):
                if self.kb_listener is not None:
                    self.kb_listener.reset_trim()

            if not raw.startswith("$IMU"):
                continue

            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 3:
                continue
            try:
                v = float(parts[1])
                deg = float(parts[2])
            except ValueError:
                continue

            with self.lock:
                self._latest_v = v
                self._latest_deg = deg
                self._latest_ts = time.time()
            self.count += 1

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


class CirReader(threading.Thread):
    def __init__(self, port, baud, out_queue, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.ser = None
        self.connected = False
        self.last_error = None

    def run(self):
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=1.0)
            self.ser.dtr = True
            self.ser.rts = True
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.connected = True
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            return

        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception as e:
                self.last_error = str(e)
                continue
            if not raw or raw.startswith("$"):
                continue
            self.out_queue.put((time.time(), raw))

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


class KeyboardListener(threading.Thread):
    def __init__(self, robot_reader, stop_event, step=3):
        super().__init__(daemon=True)
        self.robot_reader = robot_reader
        self.stop_event = stop_event
        self.step = step
        self.trim_l = 0
        self.trim_r = 0
        self.is_running = True
        self.lock = threading.Lock()
        self.is_tty = sys.stdin.isatty()
        self._old_settings = None

    def reset_trim(self):
        with self.lock:
            self.trim_l = 0
            self.trim_r = 0

    def cleanup(self):
        if self._old_settings and self.is_tty:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def run(self):
        if not self.is_tty:
            return
        import termios
        import tty
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            return

        try:
            while not self.stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                chars = sys.stdin.read(1)
                if not chars:
                    continue
                ch = chars[0]

                send_cmd = None
                updated_trim = False
                with self.lock:
                    if ch in ("s", "S"):
                        self.is_running = False
                        if self.robot_reader:
                            self.robot_reader.send_stop()
                        self.stop_event.set()
                        break
                    elif ch == " ":
                        if self.is_running:
                            self.is_running = False
                            send_cmd = "STOP"
                        else:
                            self.is_running = True
                            send_cmd = "RESUME"
                    elif ch in ("b", "B"):
                        self.is_running = True
                        send_cmd = "TURN"
                    elif ch in ("u", "U"):
                        self.trim_r = 0
                        self.trim_l += self.step
                        updated_trim = True
                    elif ch in ("j", "J"):
                        self.trim_r = 0
                        self.trim_l -= self.step
                        updated_trim = True
                    elif ch in ("o", "O"):
                        self.trim_l = 0
                        self.trim_r += self.step
                        updated_trim = True
                    elif ch in ("l", "L"):
                        self.trim_l = 0
                        self.trim_r -= self.step
                        updated_trim = True
                    elif ch in ("k", "K"):
                        self.trim_l = 0
                        self.trim_r = 0
                        updated_trim = True
                    tl, tr = self.trim_l, self.trim_r

                if self.robot_reader:
                    if send_cmd == "STOP":
                        self.robot_reader.send_stop()
                    elif send_cmd == "RESUME":
                        self.robot_reader.send_resume()
                    elif send_cmd == "TURN":
                        self.robot_reader.send_turn()
                    elif updated_trim:
                        self.robot_reader.send_trim(tl, tr)
        finally:
            self.cleanup()


class SimReaderThread(threading.Thread):
    def __init__(self, csv_file: str, fps: float, out_queue: Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.csv_file = str(resolve_file(csv_file))
        self.fps = fps
        self.queue = out_queue
        self.stop_event = stop_event
        self.connected = True
        self.last_error = None
        self._cur_v = 0.0
        self._cur_deg = 0.0
        self.lock = threading.Lock()

    def get_latest(self):
        with self.lock:
            return self._cur_v, self._cur_deg, time.time()

    def run(self):
        if not Path(self.csv_file).exists():
            self.last_error = f"File {self.csv_file} không tồn tại!"
            self.connected = False
            return
        import pandas as pd
        df = pd.read_csv(self.csv_file)
        dt_sleep = 1.0 / max(1.0, self.fps)
        total_rows = len(df)
        row_idx = 0

        while not self.stop_event.is_set():
            row = df.iloc[row_idx]
            row_idx = (row_idx + 1) % total_rows
            with self.lock:
                self._cur_v = float(row.get("robot_V", 0.0)) if pd.notnull(row.get("robot_V")) else 0.0
                self._cur_deg = float(row.get("robot_deg", 0.0)) if pd.notnull(row.get("robot_deg")) else 0.0

            aid = str(row["anchor_id"]).strip().lower()
            rng = str(row["range"])
            fp_idx = str(row.get("FP_IDX", 740))
            fp1 = str(row.get("FP_AMP1", 10000))
            fp2 = str(row.get("FP_AMP2", 10000))
            fp3 = str(row.get("FP_AMP3", 10000))
            std_n = str(row.get("STDEV_NOISE", 80))
            pwr = str(row.get("CIR_PWR", 12000))
            max_n = str(row.get("MAX_NOISE", 1000))
            cir_vals = [str(int(row[f"CIR{i}"])) for i in range(N_CIR)]
            meta = [aid, rng, fp_idx, fp1, fp2, fp3, std_n, pwr, max_n]
            line = ",".join(meta + cir_vals)
            self.queue.put((time.time(), line))
            time.sleep(dt_sleep)


def solve_linear_lse(anchor_keys, anchors, ranges):
    a0 = anchors[anchor_keys[0]]
    r0 = ranges[anchor_keys[0]]
    A = []
    b = []
    for k in anchor_keys[1:]:
        ak = anchors[k]
        rk = ranges[k]
        A.append([2.0 * (ak[0] - a0[0]), 2.0 * (ak[1] - a0[1])])
        b.append((rk**2 - r0**2) - (ak[0]**2 - a0[0]**2) - (ak[1]**2 - a0[1]**2))
    A = np.array(A)
    b = np.array(b)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, -b, rcond=None)
        return sol
    except Exception:
        return np.mean([anchors[k] for k in anchor_keys], axis=0)


def solve_lse(anchor_keys, anchors, ranges, init_pos=None):
    if len(anchor_keys) < 3:
        return None
    if init_pos is None or np.isnan(init_pos).any():
        init_pos = solve_linear_lse(anchor_keys, anchors, ranges)
    if not HAS_SCIPY:
        return init_pos
    def residuals(p):
        return [np.linalg.norm(p - anchors[k]) - ranges[k] for k in anchor_keys]
    try:
        res = least_squares(residuals, init_pos, method="lm", max_nfev=35)
        return res.x
    except Exception:
        return init_pos


class TinyCNNNumpyEngine:
    """Bộ suy luận Pure NumPy Tiny 1D-CNN (không cần TensorFlow)."""
    def __init__(self, npz_path=None):
        if npz_path is None:
            npz_path = resolve_file("fgo_ekf_weights_numpy.npz")
        data = np.load(npz_path)
        self.scaler_mean = data["scaler_mean"]
        self.scaler_scale = data["scaler_scale"]
        self.w_conv1 = data["w_conv1"]
        self.b_conv1 = data["b_conv1"]
        self.gamma_bn1 = data["gamma_bn1"]
        self.beta_bn1 = data["beta_bn1"]
        self.mean_bn1 = data["mean_bn1"]
        self.var_bn1 = data["var_bn1"]
        self.w_conv2 = data["w_conv2"]
        self.b_conv2 = data["b_conv2"]
        self.gamma_bn2 = data["gamma_bn2"]
        self.beta_bn2 = data["beta_bn2"]
        self.mean_bn2 = data["mean_bn2"]
        self.var_bn2 = data["var_bn2"]
        self.w_dense = data["w_dense"]
        self.b_dense = data["b_dense"]
        self.w_bias = data["w_bias"]
        self.b_bias = data["b_bias"]
        self.has_cls = "w_cls" in data and np.any(data["w_cls"] != 0)
        if self.has_cls:
            self.w_cls = data["w_cls"]
            self.b_cls = data["b_cls"]
        else:
            self.w_cls = None
            self.b_cls = None

    @staticmethod
    def _conv1d_same(x, w, b):
        L, C_in = x.shape
        K, _, C_out = w.shape
        pad_left = (K - 1) // 2
        pad_right = K - 1 - pad_left
        x_pad = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="constant")
        out = np.zeros((L, C_out), dtype=np.float32)
        for i in range(L):
            patch = x_pad[i:i+K]
            out[i] = np.sum(patch[:, :, None] * w, axis=(0, 1)) + b
        return out

    @staticmethod
    def _maxpool1d(x, pool_size=2):
        L, C = x.shape
        new_L = L // pool_size
        return x[:new_L*pool_size].reshape(new_L, pool_size, C).max(axis=1)

    @staticmethod
    def _batchnorm(x, gamma, beta, mean, var, eps=1e-3):
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def predict(self, feat_57d_raw):
        x = (feat_57d_raw - self.scaler_mean) / self.scaler_scale
        x = x.reshape(57, 1).astype(np.float32)
        x = self._conv1d_same(x, self.w_conv1, self.b_conv1)
        x = self._batchnorm(x, self.gamma_bn1, self.beta_bn1, self.mean_bn1, self.var_bn1)
        x = np.maximum(x, 0)
        x = self._maxpool1d(x, 2)
        x = self._conv1d_same(x, self.w_conv2, self.b_conv2)
        x = self._batchnorm(x, self.gamma_bn2, self.beta_bn2, self.mean_bn2, self.var_bn2)
        x = np.maximum(x, 0)
        x = self._maxpool1d(x, 2)
        flat = x.flatten()
        feat = np.maximum(flat @ self.w_dense + self.b_dense, 0)
        bias_hat = float((feat @ self.w_bias + self.b_bias)[0])
        if self.has_cls and self.w_cls is not None:
            probs = self._softmax(feat @ self.w_cls + self.b_cls)
            return bias_hat, probs
        return bias_hat, None
