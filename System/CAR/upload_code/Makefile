# --- Cấu hình Dự án ---
SKETCH = .
TARGET = upload_code
# Đưa FQBN về nguyên bản, bỏ upload_method đi
FQBN = STMicroelectronics:stm32:GenF1:pnum=BLUEPILL_F103C8,usb=CDCgen
BUILD_DIR = build
CPUTAPID ?= 0x2ba01477

# --- Các Targets ---
.PHONY: all compile flash clean

all: compile

compile:
	@echo "Đang biên dịch dự án..."
	# Thêm cờ --output-dir để điều hướng file thực thi vào thư mục build/
	arduino-cli compile --fqbn $(FQBN) --output-dir $(BUILD_DIR) $(SKETCH)

flash: compile
	@echo "Đang gọi OpenOCD để nạp code qua ST-Link..."
	# Lấy file .elf (chứa sẵn thông tin địa chỉ flash) đẩy thẳng xuống STM32
	openocd -f interface/stlink.cfg -c "set CPUTAPID $(CPUTAPID)" -f target/stm32f1x.cfg -c "program $(BUILD_DIR)/$(TARGET).ino.elf verify reset exit"

clean:
	@echo "Dọn dẹp..."
	arduino-cli cache clean
	rm -rf $(BUILD_DIR)