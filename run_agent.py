#!/usr/bin/env python3
"""
run_agent.py
============
Entry point tiện lợi để khởi chạy IPS-Agent Terminal:
    python3 run_agent.py
"""
import sys
from pathlib import Path

# Đảm bảo PYTHONPATH trỏ tới thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ips_agent import main

if __name__ == "__main__":
    main()
