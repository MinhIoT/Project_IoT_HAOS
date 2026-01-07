# =====================================================
# CLEAR STATIC IMAGES
# Chạy file -> xóa toàn bộ ảnh -> thoát
# =====================================================

import os

from web import UNKNOWN_DIR

CAPTURE_DIR = "static/captures"
YOLO_DIR = "static/yolo_captures"
FACE_DIR = "static/face_captures"
UNKNOWN = "static/unknown_captures"

ALLOWED_EXT = (".jpg", ".jpeg", ".png")

def clear_folder(folder):
    if not os.path.exists(folder):
        print(f"[SKIP] {folder} không tồn tại")
        return 0

    count = 0
    for f in os.listdir(folder):
        if f.lower().endswith(ALLOWED_EXT):
            os.remove(os.path.join(folder, f))
            count += 1

    print(f"[OK] Đã xóa {count} ảnh trong {folder}")
    return count


if __name__ == "__main__":
    print("=== CLEAR STATIC IMAGES ===")

    total = 0
    total += clear_folder(CAPTURE_DIR)
    total += clear_folder(FACE_DIR)
    total += clear_folder(YOLO_DIR)
    total += clear_folder(UNKNOWN)

    print(f"=== DONE | Tổng ảnh đã xóa: {total} ===")
