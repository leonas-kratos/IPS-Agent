#!/usr/bin/env python3
"""
realtime_positioning.py
=======================
Hệ thống định vị UWB thời gian thực trực tiếp trên Robot (Online FGO-EKF & Tiny 1D-CNN):
  - Tối ưu hóa đặc biệt cho Raspberry Pi 4 / ARM Linux:
      • Không yêu cầu cài đặt TensorFlow (tránh lỗi cồng kềnh, thiếu driver, tốn RAM).
      • Sử dụng Pure NumPy Inference Engine (khởi động 0.05s, chạy siêu nhẹ, RAM < 20MB).
      • Vẫn hỗ trợ TFLite và Keras nếu máy có cài đặt.
  - Đọc đồng thời 2 cổng Serial:
      1) Cổng CIR   : Mặc định /dev/ttyACM1 @ 921600 baud (DWM1001)
      2) Cổng Robot : Mặc định /dev/ttyACM0 @ 115200 baud (STM32 phát $IMU,<V>,<deg>)
      • Tự động quét và nhận diện cổng (auto_detect_ports).
  - Điều khiển xe trực tiếp bằng bàn phím (Keyboard Control):
      • [Space]   : DỪNG XE / TIẾP TỤC CHẠY (Tạm dừng)
      • [b] / [B] : QUAY TRÁI 90°
      • [s] / [S] : DỪNG HOÀN TOÀN & LƯU FILE CSV
      • [u/j, o/l]: Cân bằng vi sai PWM bánh trái / phải (TRIM)
      • [k]       : Reset TRIM về 0
  - Xử lý Deep Learning Online (57D):
      • Vector 57 đặc trưng: 50 CIR + 7 PHY (CIR_PWR, FP_AMP1..3, STDEV_NOISE, MAX_NOISE, range)
      • Tiny 1D-CNN (<10k tham số, models/fgo_ekf_weights_numpy.npz):
          - Ước lượng Range Bias b̂_i theo thời gian thực (Huber loss)
          - Phân loại trạng thái kênh (LOS, NLOS Board, NLOS Table) & trích xuất p_LOS
  - Tầng ước lượng tọa độ FGO-EKF (Sensor Fusion):
      • Prediction: Tích hợp Odometry bánh xe (robot_V) + IMU (robot_deg)
      • Update: Cập nhật cự ly hiệu chỉnh d_corr = max(10, d_raw - b̂_i) kết hợp kiểm định Chi-bình phương (NIS Gating)
  - Lưu trữ CSV trực tuyến chuẩn hóa:
      timestamp, robot_V, robot_deg, anchor_id, range, FP_IDX, FP_AMP1..3,
      STDEV_NOISE, CIR_PWR, MAX_NOISE, CIR0..CIR49, bias_hat, range_corr,
      class_pred, p_los, x_lse_raw, y_lse_raw, x_lse_ai, y_lse_ai, x_fgo_ekf, y_fgo_ekf
"""

import argparse
import csv
from datetime import datetime
import glob
import os
from pathlib import Path
from queue import Queue, Empty
import re
import select
import signal
import sys
import threading
import time

import numpy as np

# Thử import scipy cho non-linear LSE (nếu không có thì fallback sang linear LSE)
try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Cấu hình hằng số mặc định
# ---------------------------------------------------------------------------
DEFAULT_CIR_PORT = "/dev/ttyACM1"
DEFAULT_CIR_BAUD = 921600
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_BAUD = 115200

N_CIR = 50
N_META_FIELDS = 9  # anchor_id, range, FP_IDX, FP_AMP1..3, STDEV_NOISE, CIR_PWR, MAX_NOISE
N_IN = 57          # 50 CIR + 7 PHY
CLASS_NAMES = ["LOS", "NLOS (Board)", "NLOS (Table)"]


# ---------------------------------------------------------------------------
# Tìm đường dẫn file thông minh
# ---------------------------------------------------------------------------
def resolve_file_path(filename: str) -> Path:
    target = Path(filename)
    if target.exists():
        return target
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "models" / filename,
        Path(__file__).resolve().parent.parent / filename,
        Path(__file__).resolve().parent.parent / "models" / filename,
        Path("models") / filename,
        Path("Code/CAR/models") / filename,
        Path("Data/Moving") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return target


# ---------------------------------------------------------------------------
# Bộ suy luận Pure NumPy Tiny 1D-CNN (Chạy siêu tốc trên Raspberry Pi 4)
# ---------------------------------------------------------------------------
class TinyCNNNumpyEngine:
    """Mô hình Tiny 1D-CNN thuần NumPy 100%, không phụ thuộc TensorFlow hay TFLite."""
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.scaler_mean = data["scaler_mean"]
        self.scaler_scale = data["scaler_scale"]

        self.w_conv1 = data["w_conv1"]  # (5, 1, 16)
        self.b_conv1 = data["b_conv1"]  # (16,)
        self.gamma_bn1 = data["gamma_bn1"]
        self.beta_bn1 = data["beta_bn1"]
        self.mean_bn1 = data["mean_bn1"]
        self.var_bn1 = data["var_bn1"]

        self.w_conv2 = data["w_conv2"]  # (3, 16, 32)
        self.b_conv2 = data["b_conv2"]  # (32,)
        self.gamma_bn2 = data["gamma_bn2"]
        self.beta_bn2 = data["beta_bn2"]
        self.mean_bn2 = data["mean_bn2"]
        self.var_bn2 = data["var_bn2"]

        self.w_dense = data["w_dense"]  # (448, 32)
        self.b_dense = data["b_dense"]  # (32,)
        self.w_bias = data["w_bias"]    # (32, 1)
        self.b_bias = data["b_bias"]    # (1,)
        self.has_cls = "w_cls" in data and np.any(data["w_cls"] != 0)
        if self.has_cls:
            self.w_cls = data["w_cls"]  # (32, 3)
            self.b_cls = data["b_cls"]  # (3,)
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
        """Đầu vào: vector 57 số thực thô. Đầu ra: bias_hat (float), probs (3,) hoặc None."""
        # 1. Z-score normalization
        x = (feat_57d_raw - self.scaler_mean) / self.scaler_scale
        x = x.reshape(57, 1).astype(np.float32)

        # 2. Block 1: Conv1D (5x1x16) -> BN -> ReLU -> MaxPool(2)
        x = self._conv1d_same(x, self.w_conv1, self.b_conv1)
        x = self._batchnorm(x, self.gamma_bn1, self.beta_bn1, self.mean_bn1, self.var_bn1)
        x = np.maximum(x, 0)
        x = self._maxpool1d(x, 2)

        # 3. Block 2: Conv1D (3x16x32) -> BN -> ReLU -> MaxPool(2)
        x = self._conv1d_same(x, self.w_conv2, self.b_conv2)
        x = self._batchnorm(x, self.gamma_bn2, self.beta_bn2, self.mean_bn2, self.var_bn2)
        x = np.maximum(x, 0)
        x = self._maxpool1d(x, 2)

        # 4. Flatten & Dense Shared
        flat = x.flatten()
        feat = np.maximum(flat @ self.w_dense + self.b_dense, 0)

        # 5. Output: Bias regression (+ Channel Classification nếu có)
        bias_hat = float((feat @ self.w_bias + self.b_bias)[0])
        if self.has_cls and self.w_cls is not None:
            probs = self._softmax(feat @ self.w_cls + self.b_cls)
            return bias_hat, probs
        return bias_hat, None


def load_ai_engine(model_arg: str, scaler_arg: str):
    """
    Tự động chọn Engine nhẹ nhất và phù hợp nhất cho thiết bị (Raspberry Pi 4 hoặc PC).
    Ưu tiên 1: Pure NumPy Engine (.npz) -> Khởi động tức thì, không cần TensorFlow!
    Ưu tiên 2: TFLite Runtime (.tflite)
    Ưu tiên 3: TensorFlow / Keras (.keras)
    """
    npz_candidate = resolve_file_path("fgo_ekf_weights_numpy.npz")
    tflite_candidate = resolve_file_path("fgo_ekf_tiny_cnn.tflite")
    keras_candidate = resolve_file_path(model_arg)

    # 1. Thử NumPy Engine (Lựa chọn tối ưu nhất cho Pi 4)
    if npz_candidate.exists():
        print(f"[+] Sử dụng Pure NumPy Engine: {npz_candidate.name} (Tối ưu cho Pi 4, không cần TF)")
        engine = TinyCNNNumpyEngine(npz_candidate)
        return engine

    # 2. Thử TFLite Engine
    if tflite_candidate.exists():
        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow.lite as tflite
                Interpreter = tflite.Interpreter

            print(f"[+] Sử dụng TFLite Engine: {tflite_candidate.name}")
            interp = Interpreter(model_path=str(tflite_candidate))
            interp.allocate_tensors()
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()

            scaler_path = resolve_file_path(scaler_arg)
            import joblib
            scaler = joblib.load(scaler_path)

            class TFLiteWrapper:
                def predict(self, raw_feat):
                    feat_scaled = scaler.transform(raw_feat.reshape(1, 57)).astype(np.float32)
                    interp.set_tensor(input_details[0]["index"], feat_scaled)
                    interp.invoke()
                    if len(output_details) == 1:
                        b = interp.get_tensor(output_details[0]["index"])
                        return float(b[0, 0]), None
                    b = interp.get_tensor(output_details[0]["index"])
                    p = interp.get_tensor(output_details[1]["index"])
                    if b.shape[-1] != 1 and p.shape[-1] == 1:
                        b, p = p, b
                    return float(b[0, 0]), p[0]

            return TFLiteWrapper()
        except Exception:
            pass

    # 3. Thử Keras Engine
    if keras_candidate.exists():
        try:
            import tensorflow as tf
            import joblib
            print(f"[+] Sử dụng TensorFlow Keras Engine: {keras_candidate.name}")
            k_model = tf.keras.models.load_model(keras_candidate, compile=False)
            scaler_path = resolve_file_path(scaler_arg)
            scaler = joblib.load(scaler_path)

            class KerasWrapper:
                def predict(self, raw_feat):
                    feat_scaled = scaler.transform(raw_feat.reshape(1, 57)).astype(np.float32)
                    preds = k_model(feat_scaled, training=False)
                    if isinstance(preds, (list, tuple)) and len(preds) > 1:
                        return float(preds[0].numpy()[0, 0]), preds[1].numpy()[0]
                    return float(preds.numpy()[0, 0]), None

            return KerasWrapper()
        except ImportError:
            pass

    raise FileNotFoundError(
        "[-] Không tìm thấy mô hình! Đã tìm: fgo_ekf_weights_numpy.npz, fgo_ekf_tiny_cnn.tflite, fgo_ekf_tiny_cnn.keras\n"
        "    Hãy chắc chắn thư mục 'models/' có chứa file fgo_ekf_weights_numpy.npz"
    )


# ---------------------------------------------------------------------------
# Đọc file cấu hình Anchor (setup.txt)
# ---------------------------------------------------------------------------
def load_anchor_setup(setup_path: Path):
    anchors = {
        "0x1001": np.array([0.0, 0.0]),
        "0x1002": np.array([0.0, 5909.0]),
        "0x1003": np.array([10236.0, 5909.0]),
        "0x1004": np.array([10236.0, 0.0]),
    }
    actual_path = resolve_file_path(str(setup_path))
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


# ---------------------------------------------------------------------------
# Tự động quét và nhận diện cổng Serial
# ---------------------------------------------------------------------------
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
    if candidates:
        print(f"[*] Cổng khả dụng: {', '.join(candidates)}")
    else:
        print("[!] Không tìm thấy cổng /dev/ttyACM* hoặc /dev/ttyUSB* nào!")
        return user_cir_port or DEFAULT_CIR_PORT, user_robot_port or DEFAULT_ROBOT_PORT

    detected_robot = user_robot_port
    detected_cir = user_cir_port

    # 1. Tìm cổng Robot ($IMU hoặc log STM32 ở baud 115200)
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

    # 2. Cổng còn lại là CIR
    if detected_robot is not None and detected_cir is None:
        remaining = [p for p in candidates if p != detected_robot]
        if remaining:
            detected_cir = remaining[0]

    # 3. Nếu chưa tìm thấy robot, tìm CIR ở baud 921600
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

    # Phân bổ mặc định nếu không dò được
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

    print(f"[+] Nhận diện cổng kết nối:")
    print(f"    • Robot (STM32, $IMU) : {detected_robot} @ {DEFAULT_ROBOT_BAUD}")
    print(f"    • CIR   (DWM1001)     : {detected_cir} @ {DEFAULT_CIR_BAUD}")

    return detected_cir, detected_robot


# ---------------------------------------------------------------------------
# Luồng đọc cổng Robot (STM32) và gửi lệnh điều khiển
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Luồng đọc cổng CIR (DWM1001 @ 921600 baud)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Bàn phím điều khiển Non-blocking
# ---------------------------------------------------------------------------
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
        self.last_action = ""
        self.is_tty = sys.stdin.isatty()
        self._old_settings = None

    def reset_trim(self):
        with self.lock:
            self.trim_l = 0
            self.trim_r = 0
            self.last_action = "Reset TRIM về 0"

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
                        self.last_action = "DỪNG HOÀN TOÀN (s/S)"
                        self.is_running = False
                        if self.robot_reader:
                            self.robot_reader.send_stop()
                        self.stop_event.set()
                        break
                    elif ch == " ":
                        if self.is_running:
                            self.is_running = False
                            self.last_action = "DỪNG XE (Space)"
                            send_cmd = "STOP"
                        else:
                            self.is_running = True
                            self.last_action = "CHẠY TIẾP (Space)"
                            send_cmd = "RESUME"
                    elif ch in ("b", "B"):
                        self.is_running = True
                        self.last_action = "QUAY TRÁI 90° (b)"
                        send_cmd = "TURN"
                    elif ch in ("u", "U"):
                        self.trim_r = 0
                        self.trim_l += self.step
                        self.last_action = f"Tăng bánh TRÁI (+{self.trim_l} PWM)"
                        updated_trim = True
                    elif ch in ("j", "J"):
                        self.trim_r = 0
                        self.trim_l -= self.step
                        self.last_action = f"Giảm bánh TRÁI ({self.trim_l:+d} PWM)"
                        updated_trim = True
                    elif ch in ("o", "O"):
                        self.trim_r = 0
                        self.trim_r += self.step
                        self.last_action = f"Tăng bánh PHẢI (+{self.trim_r} PWM)"
                        updated_trim = True
                    elif ch in ("l", "L"):
                        self.trim_r = 0
                        self.trim_r -= self.step
                        self.last_action = f"Giảm bánh PHẢI ({self.trim_r:+d} PWM)"
                        updated_trim = True
                    elif ch in ("k", "K"):
                        self.trim_l = 0
                        self.trim_r = 0
                        self.last_action = "Reset TRIM về 0"
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


# ---------------------------------------------------------------------------
# Luồng Replay Giả lập
# ---------------------------------------------------------------------------
class SimReaderThread(threading.Thread):
    def __init__(self, csv_file: str, fps: float, out_queue: Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.csv_file = str(resolve_file_path(csv_file))
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


# ---------------------------------------------------------------------------
# Thuật toán định vị LSE & FGO-EKF
# ---------------------------------------------------------------------------
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


def solve_nonlinear_lse(anchor_keys, anchors, ranges, init_pos=None):
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


class OnlineFGOEKF:
    """FGO-EKF Sensor Fusion thời gian thực (tích hợp Odometry V và IMU deg)."""
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


# ---------------------------------------------------------------------------
# Giao diện đồ họa Matplotlib (Tùy chọn)
# ---------------------------------------------------------------------------
class RealtimeGUI:
    def __init__(self, anchors):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.anchors = anchors
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6.5))
        self.fig.canvas.manager.set_window_title("Realtime UWB + Robot Sensor Fusion (FGO-EKF)")

        xs = [p[0] for p in anchors.values()]
        ys = [p[1] for p in anchors.values()]
        self.ax.scatter(xs, ys, c="red", marker="^", s=150, zorder=5, label="Anchor")
        for aid, p in anchors.items():
            self.ax.text(p[0] + 150, p[1] + 150, f"{aid}\n({p[0]:.0f}, {p[1]:.0f})",
                         fontsize=9, weight="bold", color="darkred")

        self.tag_scatter, = self.ax.plot([], [], "bo", markersize=10, label="Tag (FGO-EKF + Robot SF)")
        self.tag_lse, = self.ax.plot([], [], "mx", markersize=8, label="Tag (AI-LSE)")
        self.traj_line, = self.ax.plot([], [], "b-", linewidth=1.5, alpha=0.6, label="Quỹ đạo FGO-EKF")
        self.heading_line, = self.ax.plot([], [], "g-", linewidth=2.5, label="Hướng Robot (Deg)")

        self.circles = {}
        for aid in anchors:
            c = plt.Circle((0, 0), 100, fill=False, linestyle="--", alpha=0.3, color="gray")
            self.ax.add_patch(c)
            self.circles[aid] = c

        margin = 1000
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("Tọa độ X (mm)", fontsize=11)
        self.ax.set_ylabel("Tọa độ Y (mm)", fontsize=11)
        self.ax.set_title("ĐỊNH VỊ REALTIME: UWB 57D + ROBOT ODOMETRY SENSOR FUSION", fontsize=12, fontweight="bold")
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper right")

        self.traj_x = []
        self.traj_y = []
        self.last_gui_time = time.time()

    def update(self, pos_ekf, pos_lse, latest_ranges, robot_deg=None):
        now = time.time()
        if now - self.last_gui_time < 0.05:
            return
        self.last_gui_time = now

        if pos_ekf is not None and not np.isnan(pos_ekf).any():
            self.tag_scatter.set_data([pos_ekf[0]], [pos_ekf[1]])
            self.traj_x.append(pos_ekf[0])
            self.traj_y.append(pos_ekf[1])
            if len(self.traj_x) > 300:
                self.traj_x.pop(0)
                self.traj_y.pop(0)
            self.traj_line.set_data(self.traj_x, self.traj_y)

            if robot_deg is not None:
                rad = -np.radians(robot_deg)
                hlen = 400.0
                hx = pos_ekf[0] + hlen * np.cos(rad)
                hy = pos_ekf[1] + hlen * np.sin(rad)
                self.heading_line.set_data([pos_ekf[0], hx], [pos_ekf[1], hy])

        if pos_lse is not None and not np.isnan(pos_lse).any():
            self.tag_lse.set_data([pos_lse[0]], [pos_lse[1]])

        for aid, r in latest_ranges.items():
            if aid in self.circles and aid in self.anchors:
                self.circles[aid].center = (self.anchors[aid][0], self.anchors[aid][1])
                self.circles[aid].set_radius(r)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Định vị thời gian thực kết hợp dữ liệu Xe Robot (Odometry+IMU) và CIR 57D (Tương thích hoàn hảo Raspberry Pi 4)."
    )
    parser.add_argument("--cir-port", type=str, default=None,
                        help="Cổng Serial CIR (mặc định: tự động nhận diện hoặc /dev/ttyACM1)")
    parser.add_argument("--cir-baud", type=int, default=DEFAULT_CIR_BAUD,
                        help=f"Baudrate cổng CIR (mặc định: {DEFAULT_CIR_BAUD})")
    parser.add_argument("--robot-port", type=str, default=None,
                        help="Cổng Serial Robot phát $IMU (mặc định: tự động nhận diện hoặc /dev/ttyACM0)")
    parser.add_argument("--robot-baud", type=int, default=DEFAULT_ROBOT_BAUD,
                        help=f"Baudrate cổng Robot (mặc định: {DEFAULT_ROBOT_BAUD})")
    parser.add_argument("--no-robot", action="store_true",
                        help="Không đọc cổng Robot (chỉ dùng UWB định vị độc lập)")
    parser.add_argument("--kp", type=float, default=None,
                        help="Hệ số KP_HEADING gửi xe robot (khuyên dùng 3.5 - 4.5)")
    parser.add_argument("--model", type=str, default="fgo_ekf_weights_numpy.npz",
                        help="Tên file mô hình (mặc định: fgo_ekf_weights_numpy.npz)")
    parser.add_argument("--scaler", type=str, default="scaler_57d.pkl",
                        help="Tên file StandardScaler")
    parser.add_argument("--setup", type=str, default="setup.txt",
                        help="Đường dẫn file tọa độ Anchor (setup.txt)")
    parser.add_argument("--height", type=float, default=0.0,
                        help="Độ cao chênh lệch Anchor và Tag H (mm, mặc định: 0.0 mm)")
    parser.add_argument("--gui", action="store_true",
                        help="Bật giao diện đồ họa 2D Matplotlib trực tiếp")
    parser.add_argument("--no-save", action="store_true",
                        help="Tắt chế độ tự động lưu file CSV log")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Đường dẫn file CSV lưu dữ liệu (mặc định: tự động sinh theo ngày giờ realtime_online_YYYYMMDD_HHMMSS.csv)")
    parser.add_argument("--sim", type=str, default=None,
                        help="Chế độ giả lập: đường dẫn file CSV để replay dữ liệu xe và UWB")
    parser.add_argument("--sim-fps", type=float, default=30.0,
                        help="Tốc độ replay giả lập (Hz, mặc định: 30.0)")

    args = parser.parse_args()

    def sig_handler(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    print("=" * 80)
    print("      HỆ THỐNG ĐỊNH VỊ REALTIME: ROBOT SENSOR FUSION + UWB 57D (FGO-EKF)")
    print("      (Tương thích tối ưu Raspberry Pi 4 - Pure NumPy Engine)")
    print("=" * 80)

    # 1. Đọc setup
    anchors = load_anchor_setup(Path(args.setup))
    anchor_keys = sorted(list(anchors.keys()))
    print("[*] Tọa độ Anchor:")
    for aid in anchor_keys:
        print(f"    • {aid}: X = {anchors[aid][0]:8.1f} mm, Y = {anchors[aid][1]:8.1f} mm")

    # 2. Tải mô hình AI Engine
    ai_engine = load_ai_engine(args.model, args.scaler)

    stop_event = threading.Event()
    cir_queue = Queue()

    sim_reader = None
    robot_reader = None
    cir_reader = None

    # 3. Kết nối cổng hoặc Giả lập
    if args.sim:
        print(f"[*] Kích hoạt chế độ GIẢ LẬP: Replay từ {args.sim} ({args.sim_fps} Hz)...")
        sim_reader = SimReaderThread(args.sim, args.sim_fps, cir_queue, stop_event)
        sim_reader.start()
        time.sleep(0.3)
    else:
        if not args.no_robot:
            cir_port, robot_port = auto_detect_ports(args.cir_port, args.robot_port)
        else:
            cir_port = args.cir_port or DEFAULT_CIR_PORT
            robot_port = None

        cir_reader = CirReader(cir_port, args.cir_baud, cir_queue, stop_event)
        cir_reader.start()
        time.sleep(0.3)
        if not cir_reader.connected:
            print(f"[-] Lỗi mở cổng CIR {cir_port}: {cir_reader.last_error}")
            stop_event.set()
            sys.exit(1)
        print(f"[+] Kết nối thành công cổng CIR: {cir_port}")

        if not args.no_robot:
            robot_reader = RobotReader(robot_port, args.robot_baud, stop_event)
            robot_reader.start()
            time.sleep(0.3)
            if not robot_reader.connected:
                print(f"[!] Cảnh báo: Không kết nối được cổng Robot {robot_port}: {robot_reader.last_error}")
                print("    Hệ thống sẽ chạy ở chế độ UWB only (không có Odometry).")
                robot_reader = None
            else:
                print(f"[+] Kết nối thành công cổng Robot: {robot_port}")

    # Bàn phím điều khiển nếu có robot
    kb_listener = None
    if robot_reader is not None:
        kb_listener = KeyboardListener(robot_reader, stop_event)
        robot_reader.kb_listener = kb_listener
        kb_listener.start()
        print("\n" + "=" * 70)
        print("  HƯỚNG DẪN ĐIỀU KHIỂN ROBOT BẰNG BÀN PHÍM:")
        print("  • [Space]   : BẮT ĐẦU CHẠY / TẠM DỪNG XE")
        print("  • [b] / [B] : QUAY TRÁI 90°")
        print("  • [s] / [S] : DỪNG HOÀN TOÀN & LƯU FILE CSV")
        print("  • [u/j, o/l]: TRIM vi sai tốc độ bánh Trái / Phải")
        print("  • [k]       : Reset TRIM về 0")
        print("=" * 70 + "\n")

    # File log CSV (Mặc định luôn tự động lưu vào CSV)
    csv_writer = None
    log_file = None
    if not args.no_save:
        out_filename = args.output or f"realtime_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = Path(out_filename)
        log_file = open(out_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(log_file)
        csv_header = (
            ["timestamp", "robot_V", "robot_deg",
             "anchor_id", "range", "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3",
             "STDEV_NOISE", "CIR_PWR", "MAX_NOISE"]
            + [f"CIR{i}" for i in range(N_CIR)]
            + ["bias_hat", "range_corr", "class_pred", "p_los",
               "x_lse_raw", "y_lse_raw", "x_lse_ai", "y_lse_ai", "x_fgo_ekf", "y_fgo_ekf"]
        )
        csv_writer.writerow(csv_header)
        log_file.flush()
        print(f"[*] Đang ghi log ra file: {out_path.resolve()}\n")

    # Khởi tạo FGO-EKF và GUI
    fgo_ekf = OnlineFGOEKF(anchors)
    gui = RealtimeGUI(anchors) if args.gui else None

    latest_raw_ranges = {}
    latest_corr_ranges = {}
    pos_lse_raw = None
    pos_lse_ai = None
    pos_fgo_ekf = None

    packet_count = 0
    cycle_count = 0
    start_time = time.time()
    last_print_time = start_time

    print("-" * 88)
    print(f"{'TIME':<11} | {'ROBOT (V, deg)':<16} | {'AID':<6} | {'RAW(mm)':<7} | {'BIAS':<6} | {'CORR(mm)':<8} | {'KÊNH':<12} | {'p_LOS':<5}")
    print("-" * 88)

    try:
        while not stop_event.is_set():
            if robot_reader is not None and robot_reader.check_done():
                print("\n[+] TÍN HIỆU TỪ XE: Robot đã hoàn thành lộ trình ($ROBOT,DONE)!")
                break

            try:
                recv_time, raw_line = cir_queue.get(timeout=0.1)
            except Empty:
                if gui:
                    cur_v, cur_deg, _ = robot_reader.get_latest() if robot_reader else (sim_reader.get_latest() if sim_reader else (0, 0, None))
                    gui.update(pos_fgo_ekf, pos_lse_ai, latest_corr_ranges, cur_deg)
                continue

            clean_line = raw_line.rstrip("|").strip()
            parts = [p.strip() for p in clean_line.split(",") if p.strip()]

            if len(parts) < N_META_FIELDS + N_CIR:
                continue

            aid = parts[0].strip().lower()
            if not aid.startswith("0x"):
                continue

            try:
                range_raw = float(parts[1])
                fp_idx = float(parts[2])
                fp_amp1 = float(parts[3])
                fp_amp2 = float(parts[4])
                fp_amp3 = float(parts[5])
                stdev_noise = float(parts[6])
                cir_pwr = float(parts[7])
                max_noise = float(parts[8])

                cir_samples = []
                for x in parts[N_META_FIELDS:N_META_FIELDS + N_CIR]:
                    try:
                        cir_samples.append(float(int(x, 16)))
                    except ValueError:
                        cir_samples.append(float(x))
            except ValueError:
                continue

            # Dữ liệu xe robot mới nhất
            if robot_reader is not None:
                cur_v, cur_deg, _ = robot_reader.get_latest()
            elif sim_reader is not None:
                cur_v, cur_deg, _ = sim_reader.get_latest()
            else:
                cur_v, cur_deg = 0.0, 0.0

            # Vector 57 đặc trưng: CIR0..CIR49 + CIR_PWR, FP_AMP1..3, STDEV_NOISE, MAX_NOISE, range
            phy_vec = [cir_pwr, fp_amp1, fp_amp2, fp_amp3, stdev_noise, max_noise, range_raw]
            feat_57d = np.array(cir_samples + phy_vec, dtype=np.float32)

            # Suy luận trực tiếp qua Pure NumPy Engine (hoặc TFLite/Keras)
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
            range_corr_3d = max(10.0, range_raw - bias_hat)
            if args.height > 0.0:
                r2d_sq = range_corr_3d**2 - args.height**2
                range_corr = float(np.sqrt(max(10.0, r2d_sq)))
            else:
                range_corr = range_corr_3d

            latest_raw_ranges[aid] = range_raw
            latest_corr_ranges[aid] = range_corr
            packet_count += 1

            # Tầng FGO-EKF Sensor Fusion
            if aid in anchors:
                fgo_ekf.predict(cur_v, cur_deg, now=recv_time)
                pos_fgo_ekf, _ = fgo_ekf.update(aid, range_corr, probs)

            # In dòng dữ liệu anchor
            t_str = datetime.fromtimestamp(recv_time).strftime("%H:%M:%S.%f")[:-4]
            robot_info_str = f"V={cur_v:4.1f}, {cur_deg:+6.1f}°"
            plos_str = f"{p_los*100:4.0f}%" if p_los is not None else "  N/A"
            print(f"{t_str:<11} | {robot_info_str:<16} | {aid:<6} | {range_raw:7.1f} | {bias_hat:+6.1f} | {range_corr:8.1f} | {state_str:<12} | {plos_str}")

            # Nghiệm LSE và tọa độ ước lượng khi có >= 3 anchors
            valid_aids = [k for k in anchor_keys if k in latest_corr_ranges]
            if len(valid_aids) >= 3:
                pos_lse_ai = solve_nonlinear_lse(valid_aids, anchors, latest_corr_ranges, init_pos=pos_lse_ai)
                pos_lse_raw = solve_nonlinear_lse(valid_aids, anchors, latest_raw_ranges, init_pos=pos_lse_raw)

                now = time.time()
                if now - last_print_time >= 0.5 or len(valid_aids) == len(anchor_keys):
                    last_print_time = now
                    cycle_count += 1
                    fps = packet_count / max(1e-3, now - start_time)

                    print("=" * 88)
                    print(f" [CHU KỲ #{cycle_count:<4} - {fps:.1f} Hz] KHOẢNG CÁCH & TỌA ĐỘ ƯỚC LƯỢNG (X, Y):")
                    if pos_lse_raw is not None:
                        print(f"   • LSE Thô (Chưa bù AI) : X = {pos_lse_raw[0]:7.1f} mm ({pos_lse_raw[0]/1000:5.2f}m) | Y = {pos_lse_raw[1]:7.1f} mm ({pos_lse_raw[1]/1000:5.2f}m)")
                    if pos_lse_ai is not None:
                        print(f"   • LSE Hiệu chỉnh AI    : X = {pos_lse_ai[0]:7.1f} mm ({pos_lse_ai[0]/1000:5.2f}m) | Y = {pos_lse_ai[1]:7.1f} mm ({pos_lse_ai[1]/1000:5.2f}m)")
                    if pos_fgo_ekf is not None:
                        print(f"   • FGO-EKF + Robot SF    : X = {pos_fgo_ekf[0]:7.1f} mm ({pos_fgo_ekf[0]/1000:5.2f}m) | Y = {pos_fgo_ekf[1]:7.1f} mm ({pos_fgo_ekf[1]/1000:5.2f}m)")
                        dist_str = " | ".join([f"{a}: {np.linalg.norm(pos_fgo_ekf - anchors[a]):.0f}mm" for a in valid_aids])
                        print(f"   • Khoảng cách Tag->Anchor: {dist_str}")
                    print("-" * 88)

            # Ghi log CSV
            if csv_writer:
                xl_raw = pos_lse_raw[0] if pos_lse_raw is not None else ""
                yl_raw = pos_lse_raw[1] if pos_lse_raw is not None else ""
                xl_ai = pos_lse_ai[0] if pos_lse_ai is not None else ""
                yl_ai = pos_lse_ai[1] if pos_lse_ai is not None else ""
                xc = pos_fgo_ekf[0] if pos_fgo_ekf is not None else ""
                yc = pos_fgo_ekf[1] if pos_fgo_ekf is not None else ""
                row_log = [
                    t_str, cur_v, cur_deg,
                    aid, range_raw, fp_idx, fp_amp1, fp_amp2, fp_amp3,
                    stdev_noise, cir_pwr, max_noise
                ] + cir_samples + [
                    bias_hat, range_corr, state_str, p_los,
                    xl_raw, yl_raw, xl_ai, yl_ai, xc, yc
                ]
                csv_writer.writerow(row_log)
                if packet_count % 10 == 0:
                    log_file.flush()

            if gui:
                gui.update(pos_fgo_ekf, pos_lse_ai, latest_corr_ranges, cur_deg)

    except (KeyboardInterrupt, SystemExit):
        print("\n\n[!] Dừng khẩn cấp từ người dùng (Ctrl+C). Đang đóng các kết nối...")
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
            print(f"[+] Đã lưu log an toàn vào file: {out_filename}")
        print("[+] Hoàn tất.")


if __name__ == "__main__":
    main()
