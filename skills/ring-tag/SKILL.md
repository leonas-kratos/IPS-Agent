---
name: ring-tag-protocol-expert
description: Kỹ năng chuyên sâu về hệ thống mạng UWB đa Tag dạng vòng Token Ring (System/RING_Tag), giao thức điều phối ss_bridge, cấu trúc frame bản tin và 5 cơ chế tự phục hồi (Self-Healing & Recovery).
version: 1.0.0
---

# UWB Multi-Tag Token Ring Protocol & Self-Healing Architecture (`System/RING_Tag/`)

## 1. Tổng quan Kiến trúc Hệ thống RING
Hệ thống `System/RING_Tag/` giải quyết bài toán **định vị đồng thời nhiều Tag (Multi-Tag Tracking)** mà không bị xung đột kênh truyền vô tuyến UWB (Collision Avoidance).
Hệ thống gồm 3 thành phần chính:
1. **Bridge Node (`ss_bridge`):** Bộ điều phối trung tâm giám sát mạng, quản lý danh sách thành viên Ring (`current_ring`), danh sách Anchor hoạt động (`current_anchor_list`), và phát lệnh tái đồng bộ.
2. **Tag Nodes (`ss_twr_init`):** Các thiết bị Tag di động (được định danh bằng `MY_TAG_ID`, ví dụ `0x0006`). Mỗi Tag chỉ phát sóng đo khoảng cách khi nhận được Token (`TOKN`) từ Tag đi trước trong Ring (`my_next_tag`).
3. **Anchor Nodes (`ss_twr_resp`):** Các trạm mốc cố định phản hồi các bản tin đo cự ly `POLL` / `RESP` từ Tag.

---

## 2. Định dạng Khung Bản Tin (Protocol Frame Headers)

Mọi khung bản tin UWB trong hệ thống đều tuân thủ chuẩn IEEE 802.15.4 với PAN ID `0xDECA`:

| Tên Bản Tin | Ký hiệu (ASCII) | Mã Hex | Chiều dài | Vai trò & Chức năng Kỹ thuật |
|:---|:---:|:---:|:---:|:---|
| **POLL** | `'I','O','V','E'` | `0xE0` | 22 bytes | Tag gửi tín hiệu yêu cầu đo cự ly tới Anchor. |
| **RESP** | `'V','E','I','O'` | `0xE1` | 28 bytes | Anchor phản hồi kèm nhãn thời gian thu phát ($T_{rx}, T_{tx}$). |
| **TOKEN** | `'T','O','K','N'` | `0xE2` | 16 bytes | Trao quyền phát sóng cho Tag tiếp theo trong chu kỳ Ring. |
| **DATA** | `'D','A','T','A'` | `0xE3` | Biến đổi | Tag gửi tổng hợp cự ly tới tất cả Anchor về Bridge. |
| **RING** | `'R','I','N','G'` | `0xE4` | Biến đổi | Bridge phát sóng cấu hình thành viên và thứ tự vòng tròn các Tag. |
| **A_HELLO** | `'A','H','L','O'` | `0xE5` | 14 bytes | Anchor mới khởi động gửi để xin gia nhập mạng. |
| **A_LIST** | `'A','L','S','T'` | `0xE6` | Biến đổi | Bridge gửi danh sách toàn bộ Anchor đang hoạt động cho các Tag. |
| **A_REMOVE** | `'A','R','M','V'` | `0xE7` | 14 bytes | Bridge thông báo loại bỏ một Anchor hỏng khỏi mạng. |
| **A_DEAD** | `'A','D','E','D'` | `0xE8` | 14 bytes | Tag chủ động báo cáo Anchor bị mất sóng về Bridge. |

---

## 3. 5 Cơ chế Phục hồi Tự động (Self-Healing & Recovery Mechanisms)

Trong mã nguồn `System/RING_Tag/dwm1001-examples-master/examples/ss_bridge/ss_init_main.c`, hệ thống được trang bị 5 tầng phục hồi lỗi:

### Cơ chế 1: Phục hồi khi Tag bị chết hoặc mất kết nối (Tag Timeout & Ring Rebuild)
- **Giám sát nhịp sống:** Bridge lưu thời điểm nhận bản tin cuối cùng của từng Tag trong `alive_table[i].last_seen_tick`.
- **Ngưỡng Timeout:** `TAG_TIMEOUT_MS = 1000` ms.
- **Hành động phục hồi:**
  1. Nếu `(now - last_seen_tick) >= 1000 ms`, hàm `rebuild_ring_if_needed()` ngay lập tức loại bỏ Tag chết khỏi mảng `current_ring`.
  2. Dồn mảng `alive_table`, giảm `current_ring_size`.
  3. Bridge kích hoạt `ring_send_wrapper()`, broadcast bản tin `RING` mới cho các Tag còn lại để vòng Token tiếp tục quay mà không bị đứt đoạn.

### Cơ chế 2: Phục hồi khi lệch nhịp Token (Rhythm Monitoring & Disorder Recovery)
- **Giám sát thứ tự:** Bridge theo dõi thứ tự các Tag phát dữ liệu qua hàm `check_rhythm(tag_id)`. Vị trí kế tiếp mong đợi là `expected_next_idx = (tag_pos + 1) % current_ring_size`.
- **Ngưỡng lệch nhịp:** `DISORDER_THRESHOLD = 50`.
- **Hành động phục hồi:**
  - Nếu các Tag phát sai thứ tự (do mất Token hoặc đụng độ sóng), biến đếm `disorder_count` tăng dần.
  - Khi `disorder_count >= 50`, Bridge kết luận vòng bị loạn nhịp (`[Bridge] Disorder → resending RING`), lập tức phát lại gói `RING` (`0xE4`) để thiết lập lại trật tự cho toàn bộ mạng.

### Cơ chế 3: Tự động loại bỏ Anchor hỏng (Dead Anchor Removal)
- **Phát hiện lỗi đo:** Khi Tag đo cự ly tới Anchor không phản hồi, giá trị trả về là `0xFFFFFFFF`.
- **Ngưỡng chu kỳ chết:** `ANCHOR_DEAD_CYCLES = 2`.
- **Hành động phục hồi (`check_anchor_health`):**
  1. Nếu TẤT CẢ các Tag đang sống đều không thể kết nối tới Anchor đó trong 2 chu kỳ liên tiếp (hoặc nhận bản tin `A_DEAD` `0xE8`), Bridge xác nhận Anchor đã hỏng.
  2. Bridge phát bản tin `A_REMOVE` (`0xE7`) thông báo cho toàn mạng gỡ bỏ Anchor này.
  3. Đánh dấu `anchor_table[idx].active = 0`, gọi `rebuild_anchor_list()`, và broadcast bản tin `A_LIST` (`0xE6`) mới để các Tag không lãng phí thời gian đo với Anchor hỏng.

### Cơ chế 4: Tự động phát hiện và Tái gia nhập Anchor (Anchor Dynamic Re-join)
- Khi một Anchor được cấp nguồn lại hoặc kết nối trở lại:
- Anchor phát sóng bản tin `A_HELLO` (`0xE5`, `anch_hello_hdr_ref`).
- Bridge nhận bản tin qua hàm `process_anch_hello()`:
  - Đánh dấu `active = 1`, xóa sạch bộ đếm `dead_cycle_count`.
  - Tự động sắp xếp lại danh sách `current_anchor_list` theo ID tăng dần (`rebuild_anchor_list()`).
  - Phát bản tin `A_LIST` (`0xE6`) cập nhật tới toàn bộ Tag để đưa Anchor này vào chu trình đo ngay lập tức.

### Cơ chế 5: Tái đồng bộ định kỳ & Chống va chạm gói tin (Resync & Cooldown)
- `RING_RESYNC_MS = 50` ms: Tần số gửi lại cấu hình Ring nếu phát hiện vẫn còn Tag chưa đồng bộ (`all_tags_ring_synced() == 0`).
- `ANCH_RESYNC_MS = 50` ms: Tần số gửi lại danh sách Anchor cho Tag chưa cập nhật.
- `RING_COOLDOWN_MS = 10` ms: Giãn cách tối thiểu giữa 2 lần phát sóng quản trị của Bridge để tránh bão nghẽn kênh UWB.
- Phía Tag (`ss_init_main.c`): `MAX_RANGING_RETRY = 5`, giãn cách retry 5 ms để bù đắp các khung hình mất mát ngẫu nhiên.
