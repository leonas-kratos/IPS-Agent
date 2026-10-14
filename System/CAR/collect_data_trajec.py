#!/usr/bin/env python3
"""
collect_cir_data_v2.py
=======================
Thu thập đồng thời:
  1) Dữ liệu CIR từ cổng serial DWM (mặc định /dev/ttyACM0, baud 921600)
  2) Dữ liệu vận tốc/góc của robot từ cổng serial IMU (mặc định /dev/ttyUSB0,
     baud 115200), dòng dạng:  $IMU,<V>,<deg>   ví dụ:  $IMU,0.00,-156.19

Hai cổng được đọc song song bằng 2 thread riêng. Mỗi khi nhận được một dòng
CIR hợp lệ, script sẽ ghép nó với giá trị (V, deg) MỚI NHẤT đã nhận được từ
cổng IMU tại thời điểm đó ("last known value"), kèm timestamp do máy tính
gắn vào lúc dòng CIR được xử lý (firmware IMU không tự gửi timestamp).

Cột CSV xuất ra:
    timestamp, robot_V, robot_deg, anchor_id, range, FP_IDX,
    FP_AMP1, FP_AMP2, FP_AMP3, STDEV_NOISE, CIR_PWR, MAX_NOISE,
    CIR0 ... CIR{n_cir-1}

Cửa sổ CIR
----------
Giống bản gốc: firmware in ra cửa sổ n_cir mẫu quanh First Path (mặc định 50).
Nếu đổi độ rộng cửa sổ trong firmware (ss_init_main.c) thì phải đổi luôn
--n-cir ở đây.
"""

import argparse
import csv
import select
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import serial

# ---------------------------------------------------------------------------
# Cấu hình mặc định
# ---------------------------------------------------------------------------
DEFAULT_CIR_PORT = "/dev/ttyACM1"
DEFAULT_CIR_BAUD = 921600
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_BAUD = 115200
DEFAULT_N_CIR = 50
# Số trường đứng trước chuỗi CIR: anchor_id, range, FP_IDX, FP_AMP1..3,
# STDEV_NOISE, CIR_PWR, MAX_NOISE
N_META_FIELDS = 9


# ---------------------------------------------------------------------------
# Thread đọc cổng IMU (robot): chỉ cập nhật giá trị (V, deg) mới nhất
# ---------------------------------------------------------------------------
class RobotReader(threading.Thread):
    """Đọc liên tục cổng serial của robot, giữ giá trị (V, deg) mới nhất,
    và điều khiển lệnh bắt đầu / dừng xe.

    Dòng dữ liệu từ robot: $IMU,<V>,<deg> hoặc $ROBOT,DONE
    """

    def __init__(self, port, baud, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self._latest_v = None
        self._latest_deg = None
        self._latest_ts = None
        self.count = 0
        self.ser = None
        self.connected = False
        self.last_error = None
        self.is_done = False  # Bật lên khi nhận tín hiệu $ROBOT,DONE
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
            time.sleep(0.3)
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

            # Phát hiện tín hiệu xe đã hoàn thành lộ trình và dừng
            if "$ROBOT,DONE" in raw or "$DONE" in raw:
                with self.lock:
                    self.is_done = True
                continue

            # Reset trim khi xe hoàn thành cạnh hoặc quay góc sang cạnh mới
            if "[NAV]" in raw and ("Canh" in raw or "Quay" in raw):
                if self.kb_listener is not None:
                    self.kb_listener.reset_trim()

            if not raw.startswith("$IMU"):
                continue

            parts = [p.strip() for p in raw.split(",")]
            # Kỳ vọng: ["$IMU", "<V>", "<deg>"]
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

        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Thread đọc cổng CIR: đẩy các dòng hợp lệ vào hàng đợi để luồng chính xử lý
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
            time.sleep(0.3)
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
            if not raw:
                continue
            # Bỏ qua các dòng không phải CIR (vd $IMU lọt vào cùng cổng, log khác)
            if raw.startswith("$"):
                continue
            # Gắn timestamp ngay khi nhận được dòng, trước khi đẩy vào hàng đợi
            self.out_queue.put((time.time(), raw))

        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Thread lắng nghe bàn phím trực tiếp:
#   u: Tăng bánh trái | j: Giảm bánh trái
#   o: Tăng bánh phải | l: Giảm bánh phải
# ---------------------------------------------------------------------------
class KeyboardListener(threading.Thread):
    """Lắng nghe phím bấm trực tiếp từ terminal (non-blocking):
    - ' ' (Space): DỪNG XE / TIẾP TỤC CHẠY
    - 'b' / 'B'  : QUAY TRÁI 90°
    - 's' / 'S'  : DỪNG HOÀN TOÀN THU MẪU & LƯU CSV
    - 'u' / 'U'  : TĂNG vận tốc bánh TRÁI (+step PWM)
    - 'j' / 'J'  : GIẢM vận tốc bánh TRÁI (-step PWM)
    - 'o' / 'O'  : TĂNG vận tốc bánh PHẢI (+step PWM)
    - 'l' / 'L'  : GIẢM vận tốc bánh PHẢI (-step PWM)
    - 'k' / 'K'  : Cân bằng lại (Reset TRIM về 0)
    """

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

    def get_status(self):
        with self.lock:
            return self.trim_l, self.trim_r, self.is_running, self.last_action

    def get_trim(self):
        with self.lock:
            return self.trim_l, self.trim_r, self.last_action

    def reset_trim(self):
        with self.lock:
            self.trim_l = 0
            self.trim_r = 0
            self.last_action = "Reset TRIM về 0"

    def cleanup(self):
        if self._old_settings and self.is_tty:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    def run(self):
        if not self.is_tty:
            return

        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            self._old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return

        try:
            while not self.stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r or self.stop_event.is_set():
                    continue

                # Đọc cạn toàn bộ buffer ký tự đang chờ để không bị trễ/ứ đọng phím cũ
                chars = []
                while True:
                    try:
                        c = sys.stdin.read(1)
                        if not c:
                            break
                        chars.append(c)
                    except Exception:
                        break
                    r_check, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if not r_check:
                        break

                if not chars:
                    continue

                if any(c in ("\x03", "\x04") for c in chars):  # Ctrl+C hoặc Ctrl+D
                    self.stop_event.set()
                    break

                # Lọc các phím điều khiển hợp lệ và lấy lệnh MỚI NHẤT được bấm
                valid_keys = [c for c in chars if c in ("u", "U", "o", "O", "j", "J", "l", "L", "k", "K", " ", "b", "B", "s", "S")]
                if not valid_keys:
                    continue
                ch = valid_keys[-1]

                updated_trim = False
                send_cmd = None
                with self.lock:
                    if ch in ("s", "S"):
                        # Dừng hoàn toàn thu mẫu
                        self.last_action = "DỪNG HOÀN TOÀN THU MẪU (s/S)"
                        self.is_running = False
                        if self.robot_reader is not None:
                            self.robot_reader.send_stop()
                        self.stop_event.set()
                        break

                    elif ch == " ":
                        # Ấn Space để dừng / tiếp tục chạy
                        if self.is_running:
                            self.is_running = False
                            self.last_action = "DỪNG XE (Space)"
                            send_cmd = "STOP"
                        else:
                            self.is_running = True
                            self.last_action = "CHẠY TIẾP (Space)"
                            send_cmd = "RESUME"

                    elif ch in ("b", "B"):
                        # Ấn b/B để quay trái 90 độ
                        self.is_running = True
                        self.last_action = "QUAY TRÁI 90° (b)"
                        send_cmd = "TURN"

                    elif ch in ("u", "U"):
                        self.trim_r = 0
                        if self.trim_l < 0:
                            self.trim_l = self.step
                        else:
                            self.trim_l += self.step
                        self.last_action = f"Tăng bánh TRÁI (+{self.trim_l} PWM)"
                        updated_trim = True

                    elif ch in ("o", "O"):
                        self.trim_l = 0
                        if self.trim_r < 0:
                            self.trim_r = self.step
                        else:
                            self.trim_r += self.step
                        self.last_action = f"Tăng bánh PHẢI (+{self.trim_r} PWM)"
                        updated_trim = True

                    elif ch in ("j", "J"):
                        self.trim_r = 0
                        self.trim_l -= self.step
                        self.last_action = f"Giảm bánh TRÁI ({self.trim_l:+d} PWM)"
                        updated_trim = True

                    elif ch in ("l", "L"):
                        self.trim_l = 0
                        self.trim_r -= self.step
                        self.last_action = f"Giảm bánh PHẢI ({self.trim_r:+d} PWM)"
                        updated_trim = True

                    elif ch in ("k", "K"):
                        self.trim_l = 0
                        self.trim_r = 0
                        self.last_action = "Reset TRIM về 0"
                        updated_trim = True

                    tl = self.trim_l
                    tr = self.trim_r

                if self.robot_reader is not None:
                    if send_cmd == "STOP":
                        self.robot_reader.send_stop()
                    elif send_cmd == "RESUME":
                        self.robot_reader.send_resume()
                    elif send_cmd == "TURN":
                        self.robot_reader.send_turn()
                    elif updated_trim:
                        self.robot_reader.send_trim(tl, tr)

        except Exception:
            pass
        finally:
            self.cleanup()


def auto_detect_ports(user_cir_port=None, user_robot_port=None):
    """
    Tự động quét và nhận diện cổng serial:
    - Thường là /dev/ttyACM* (hoặc /dev/ttyUSB*)
    - Cổng gửi định dạng $IMU (baud 115200) là xe (Robot)
    - Cổng còn lại là CIR (baud 921600)
    """
    if user_cir_port is not None and user_robot_port is not None:
        return user_cir_port, user_robot_port

    import glob
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not candidates:
        import serial.tools.list_ports
        candidates = sorted([p.device for p in serial.tools.list_ports.comports() if not p.device.startswith("/dev/ttyS")])

    print("[*] Đang tự động quét và nhận diện cổng serial...")
    if candidates:
        print(f"[*] Các cổng tìm thấy: {', '.join(candidates)}")
    else:
        print("[!] Không tìm thấy cổng /dev/ttyACM* hoặc /dev/ttyUSB* nào!")
        return user_cir_port or DEFAULT_CIR_PORT, user_robot_port or DEFAULT_ROBOT_PORT

    detected_robot = user_robot_port
    detected_cir = user_cir_port

    # 1. Tìm cổng Robot bằng định dạng $IMU hoặc log khởi động STM32 ở baud 115200
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
                    while time.time() - start < 1.0:
                        line = ser.readline().decode("utf-8", errors="ignore")
                        if any(k in line for k in ["$IMU", "STM32", "Robot", "BNO", "NAV", "Zero raw"]):
                            detected_robot = port
                            break
            except Exception:
                continue
            if detected_robot is not None:
                break

    # 2. "Xe có định dạng $IMU còn CIR thì là cái còn lại"
    if detected_robot is not None and detected_cir is None:
        remaining = [p for p in candidates if p != detected_robot]
        if remaining:
            detected_cir = remaining[0]

    # 3. Dự phòng: nếu chưa tìm thấy robot, kiểm tra CIR ở baud 921600 (phát 0x... Anchor ID)
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
                    while time.time() - start < 1.0:
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

    # Phân bổ mặc định: Robot là ACM0, CIR là ACM1 (hoặc cổng còn lại)
    if detected_robot is None and detected_cir is None:
        if len(candidates) >= 2:
            detected_robot = candidates[0]  # /dev/ttyACM0 là Robot
            detected_cir = candidates[1]    # /dev/ttyACM1 là CIR
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

    print(f"[+] Nhận diện cổng thành công:")
    print(f"    - Robot (STM32, $IMU) : {detected_robot}")
    print(f"    - CIR   (DWM1001)     : {detected_cir}")

    return detected_cir, detected_robot


def get_user_inputs():
    print("=" * 65)
    print("        CHƯƠNG TRÌNH THU THẬP DỮ LIỆU CIR + QUỸ ĐẠO ROBOT")
    print("   [Space]: Dừng/Chạy tiếp | [b]: Quay trái | [s]: Dừng thu mẫu")
    print("=" * 65)

    default_filename = f"cir_trajec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_input = input(f">> Nhập tên file lưu (Enter để mặc định: Data/{default_filename}): ").strip()
    if not out_input:
        output_path = Path("Data") / default_filename
    else:
        output_path = Path(out_input)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Thu thập đồng thời dữ liệu CIR và dữ liệu robot (V, deg) từ 2 cổng serial, lưu chung 1 CSV."
    )
    parser.add_argument("--kp", type=float, default=None,
                        help="Hệ số KP_HEADING gửi tới xe (khuyên dùng 3.5 - 4.5, mặc định 3.8)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Đường dẫn file CSV lưu dữ liệu")
    parser.add_argument("-n", "--samples", type=int, default=None,
                        help="Giới hạn số lượng mẫu CIR tối đa (tùy chọn, mặc định: dừng khi bấm s/S)")

    parser.add_argument("--cir-port", type=str, default=None,
                        help="Cổng serial CIR (mặc định: tự động nhận diện)")
    parser.add_argument("--cir-baud", type=int, default=DEFAULT_CIR_BAUD,
                        help=f"Baudrate cổng CIR (mặc định: {DEFAULT_CIR_BAUD})")
    parser.add_argument("--robot-port", type=str, default=None,
                        help="Cổng serial robot/IMU (mặc định: tự động nhận diện cổng phát $IMU)")
    parser.add_argument("--robot-baud", type=int, default=DEFAULT_ROBOT_BAUD,
                        help=f"Baudrate cổng robot/IMU (mặc định: {DEFAULT_ROBOT_BAUD})")
    parser.add_argument("--no-robot", action="store_true",
                        help="Bỏ qua việc đọc cổng robot/IMU (chỉ thu CIR, cột robot sẽ để trống)")

    parser.add_argument("--n-cir", type=int, default=DEFAULT_N_CIR,
                        help=f"Số mẫu CIR mỗi dòng, PHẢI khớp firmware (mặc định: {DEFAULT_N_CIR})")
    parser.add_argument("--save-hex", action="store_true",
                        help="Lưu giá trị CIR ở dạng chuỗi HEX thay vì tự động chuyển sang số thập phân")
    parser.add_argument("--trim-step", type=int, default=3,
                        help="Độ tăng/giảm PWM mỗi lần bấm phím j hoặc l (mặc định: 3 PWM)")
    parser.add_argument("--max-robot-age", type=float, default=2.0,
                        help="Số giây tối đa cho phép giá trị robot 'cũ' trước khi coi là mất kết nối (chỉ cảnh báo, không chặn ghi) (mặc định: 2.0)")
    args = parser.parse_args()
    n_cir = args.n_cir

    csv_header = (
        ["timestamp", "robot_V", "robot_deg",
         "anchor_id", "range", "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3",
         "STDEV_NOISE", "CIR_PWR", "MAX_NOISE"]
        + [f"CIR{i}" for i in range(n_cir)]
    )

    if args.output:
        output_file = args.output
    else:
        output_file = get_user_inputs()

    total_samples = args.samples if args.samples is not None else 99999999

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tự động nhận diện cổng CIR và Robot
    if not args.no_robot:
        cir_port, robot_port = auto_detect_ports(args.cir_port, args.robot_port)
    else:
        cir_port = args.cir_port or DEFAULT_CIR_PORT
        robot_port = None

    print("\n" + "-" * 60)
    print(f"[*] Cổng CIR       : {cir_port} @ {args.cir_baud} baud")
    if not args.no_robot:
        print(f"[*] Cổng Robot/IMU : {robot_port} @ {args.robot_baud} baud")
    else:
        print("[*] Cổng Robot/IMU : (bỏ qua, --no-robot)")
    print(f"[*] Số mẫu CIR/dòng: {n_cir}")
    print(f"[*] Chế độ điều khiển:")
    print(f"    • [Space]   : DỪNG XE / TIẾP TỤC CHẠY (Tạm dừng)")
    print(f"    • [b] / [B] : QUAY TRÁI 90°")
    print(f"    • [s] / [S] : DỪNG HOÀN TOÀN THU MẪU & LƯU FILE CSV")
    print(f"    • [u/j, o/l]: Bù vận tốc bánh trái / phải (TRIM)")
    print(f"    • [k]       : Reset TRIM về 0")
    if total_samples < 900000:
        print(f"[*] Giới hạn mẫu   : {total_samples:,} mẫu")
    else:
        print(f"[*] Chế độ dừng    : Bấm phím [s] hoặc [S] để dừng hoàn toàn")
    print(f"[*] File xuất      : {out_path.resolve()}")
    print("-" * 60 + "\n")

    stop_event = threading.Event()
    cir_queue = Queue()

    cir_reader = CirReader(cir_port, args.cir_baud, cir_queue, stop_event)
    cir_reader.start()
    time.sleep(0.5)
    if not cir_reader.connected:
        print(f"[-] Lỗi khi mở cổng CIR {cir_port}: {cir_reader.last_error}")
        stop_event.set()
        sys.exit(1)
    print(f"[+] Đã kết nối cổng CIR: {cir_port}")

    robot_reader = None
    if not args.no_robot:
        robot_reader = RobotReader(robot_port, args.robot_baud, stop_event)
        robot_reader.start()
        time.sleep(0.5)
        if not robot_reader.connected:
            print(f"[-] Lỗi khi mở cổng Robot/IMU {robot_port}: {robot_reader.last_error}")
            print("[-] Dừng chương trình. Dùng --no-robot nếu muốn thu CIR mà không cần robot.")
            stop_event.set()
            sys.exit(1)
        print(f"[+] Đã kết nối cổng Robot/IMU: {robot_port}")

    f = open(out_path, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(csv_header)

    # Gửi lệnh bắt đầu tới robot sau khi mọi kết nối và file đã sẵn sàng
    if robot_reader is not None:
        print("[*] Đang gửi lệnh CMD,START tới robot để bắt đầu di chuyển...")
        time.sleep(0.2)
        robot_reader.send_start(kp=args.kp)
        if args.kp is not None:
            print(f"[+] Đã kích hoạt xe chạy thẳng thành công (KP = {args.kp})!")
        else:
            print(f"[+] Đã kích hoạt xe chạy thẳng thành công!")

    kb_listener = None
    if robot_reader is not None:
        kb_listener = KeyboardListener(robot_reader, stop_event, step=args.trim_step)
        robot_reader.kb_listener = kb_listener
        kb_listener.start()
        print("\n" + "=" * 68)
        print("  HƯỚNG DẪN ĐIỀU KHIỂN BÀN PHÍM:")
        print("  • Phím [Space]     : DỪNG XE / TIẾP TỤC CHẠY (Tạm dừng)")
        print("  • Phím [b] hoặc [B]: QUAY TRÁI 90°")
        print("  • Phím [s] hoặc [S]: DỪNG HOÀN TOÀN THU MẪU & LƯU FILE CSV")
        print(f"  • BÁNH TRÁI        : Phím [u] = TĂNG (+{args.trim_step})  |  Phím [j] = GIẢM (-{args.trim_step})")
        print(f"  • BÁNH PHẢI        : Phím [o] = TĂNG (+{args.trim_step})  |  Phím [l] = GIẢM (-{args.trim_step})")
        print("  • Phím [k]         : CÂN BẰNG LẠI (Reset TRIM về 0)")
        print("  • Phím [Ctrl+C]    : Dừng khẩn cấp và lưu file CSV")
        print("=" * 68)

    collected_count = 0
    anchor_counter = Counter()
    start_time = time.time()
    last_ui_time = start_time
    recent_count = 0
    speed_sps = 0.0

    expected_cols = N_META_FIELDS + n_cir
    bad_lens = Counter()
    warned = False
    stale_robot_warned = False

    print("\n[*] Đang ghi nhận dữ liệu... (Nhấn [s] để kết thúc thu mẫu hoặc [Space] để dừng xe)\n")

    try:
        while collected_count < total_samples:
            if stop_event.is_set():
                print("\n\n[+] Người dùng đã yêu cầu dừng hoàn toàn thu mẫu!")
                break

            # Kiểm tra xem robot đã báo hoàn thành lộ trình chưa
            if robot_reader is not None and robot_reader.check_done():
                print("\n\n[+] TÍN HIỆU TỪ XE: Robot đã hoàn thành lộ trình và dừng lại ($ROBOT,DONE)!")
                print("[*] Tự động dừng thu thập mẫu.")
                break

            try:
                recv_ts, raw_line = cir_queue.get(timeout=0.1)
            except Empty:
                if stop_event.is_set():
                    break
                continue

            clean_line = raw_line.rstrip("|").strip()
            parts = [p.strip() for p in clean_line.split(",") if p.strip()]

            if len(parts) != expected_cols:
                bad_lens[len(parts)] += 1
                if not warned and sum(bad_lens.values()) >= 5:
                    warned = True
                    seen = ", ".join(f"{k} trường ×{v}" for k, v in bad_lens.most_common(3))
                    print(f"\n[!] CẢNH BÁO: dòng nhận được không khớp định dạng "
                          f"({seen}; cần đúng {expected_cols} trường).")
                    print(f"    → kiểm tra firmware có in đúng {n_cir} mẫu CIR "
                          f"quanh FP không, hoặc chỉnh lại --n-cir.")
                continue

            anchor_id = parts[0].strip().lower()
            if not (anchor_id.startswith("0x") or anchor_id.startswith("0X")):
                continue

            if not args.save_hex:
                try:
                    cir_vals = [int(x, 16) for x in parts[N_META_FIELDS:]]
                    meta_row = parts[:N_META_FIELDS] + cir_vals
                except ValueError:
                    continue
            else:
                meta_row = parts

            # Lấy giá trị robot mới nhất tại thời điểm này
            robot_v, robot_deg, robot_ts = (None, None, None)
            if robot_reader is not None:
                robot_v, robot_deg, robot_ts = robot_reader.get_latest()
                if robot_ts is not None:
                    age = recv_ts - robot_ts
                    if age > args.max_robot_age and not stale_robot_warned:
                        stale_robot_warned = True
                        print(f"\n[!] CẢNH BÁO: dữ liệu robot bị trễ {age:.1f}s so với mẫu CIR "
                              f"(kiểm tra kết nối cổng {args.robot_port}).")
                elif not stale_robot_warned:
                    # chưa nhận được dòng $IMU nào
                    pass

            ts_str = datetime.fromtimestamp(recv_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row = [ts_str, robot_v if robot_v is not None else "",
                   robot_deg if robot_deg is not None else ""] + meta_row

            writer.writerow(row)

            collected_count += 1
            recent_count += 1
            anchor_counter[anchor_id] += 1

            now = time.time()
            if now - last_ui_time >= 0.1 or collected_count == total_samples:
                elapsed = now - start_time
                dt = now - last_ui_time
                if dt > 0:
                    speed_sps = recent_count / dt
                recent_count = 0
                last_ui_time = now

                anchors_stat = " | ".join(f"{k}: {v}" for k, v in sorted(anchor_counter.items()))
                robot_stat = f" | V={robot_v:.2f} deg={robot_deg:.2f}" if robot_v is not None else " | V=-- deg=--"

                trim_stat = ""
                car_stat = ""
                if kb_listener is not None:
                    trim_l, trim_r, is_running, _ = kb_listener.get_status()
                    car_stat = " | Xe: [CHẠY]" if is_running else " | Xe: [DỪNG]"
                    if trim_l != 0 or trim_r != 0:
                        trim_stat = f" | TRIM:[L={trim_l:+d}, R={trim_r:+d}]"
                    else:
                        trim_stat = " | TRIM:0"

                if total_samples < 900000:
                    progress = (collected_count / total_samples) * 100
                    bar_len = 20
                    filled = int(bar_len * collected_count / total_samples)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    eta = (total_samples - collected_count) / speed_sps if speed_sps > 0 else 0

                    sys.stdout.write(
                        f"\r[{bar}] {collected_count}/{total_samples} ({progress:5.1f}%) "
                        f"| Tốc độ: {speed_sps:5.1f} mẫu/s "
                        f"| Đã chạy: {elapsed:4.1f}s | ETA: {eta:4.1f}s "
                        f"{car_stat}{robot_stat}{trim_stat} | [{anchors_stat}]"
                    )
                else:
                    sys.stdout.write(
                        f"\r[ĐANG THU] Đã thu: {collected_count:,} mẫu "
                        f"| Tốc độ: {speed_sps:5.1f} mẫu/s "
                        f"| Đã chạy: {elapsed:4.1f}s "
                        f"{car_stat}{robot_stat}{trim_stat} | [{anchors_stat}]"
                    )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[!] Người dùng đã bấm Ctrl+C để dừng sớm!")
        if robot_reader is not None:
            print("[*] Đang gửi lệnh dừng khẩn cấp CMD,STOP tới robot...")
            robot_reader.send_stop()
    finally:
        stop_event.set()
        if robot_reader is not None:
            robot_reader.send_stop()
        if kb_listener is not None:
            kb_listener.cleanup()
        f.flush()
        f.close()

    total_time = time.time() - start_time
    avg_speed = collected_count / total_time if total_time > 0 else 0

    print("\n\n" + "=" * 60)
    print("                     KẾT QUẢ THU THẬP")
    print("=" * 60)
    print(f"Tổng số mẫu lưu được : {collected_count:,} mẫu")
    print(f"Tổng thời gian       : {total_time:.2f} giây")
    print(f"Tốc độ trung bình    : {avg_speed:.1f} mẫu/giây (sps)")
    if robot_reader is not None:
        print(f"Số dòng IMU nhận được: {robot_reader.count:,}")
    print(f"Phân bố theo Anchor  :")
    for aid, cnt in sorted(anchor_counter.items()):
        print(f"  - {aid}: {cnt:,} mẫu ({cnt/collected_count*100:.1f}%)" if collected_count else f"  - {aid}: 0")
    print(f"Đường dẫn file CSV   : {out_path.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
