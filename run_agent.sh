#!/usr/bin/env bash
# Script 1-Click khởi chạy IPS-UWB AI Agent Terminal

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Kiểm tra python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "❌ Lỗi: Máy tính của bạn chưa cài đặt Python!"
    exit 1
fi

# Chạy trực tiếp agent
$PY_CMD ips_agent.py "$@"
