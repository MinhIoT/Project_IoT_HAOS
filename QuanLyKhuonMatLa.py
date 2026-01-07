# =====================================================
# MODULE: QuanLyKhuonMatLa
#
# Chức năng:
# - Lọc điều kiện trước khi lưu khuôn mặt lạ
# - Tránh spam ảnh (giới hạn tần suất)
# - Gom nhóm ảnh theo thời gian xuất hiện
# - Tự động xóa ảnh theo số lượng và thời gian tồn tại
# - Tạm khóa lưu ảnh sau khi gán danh tính
# =====================================================

import os
import time
import cv2
from datetime import datetime

class UnknownFaceManager:
    def __init__(
        self,
        save_dir,
        min_confidence=0.5,
        min_face_size=80,
        save_interval=3.0,
        group_interval=5.0,
        freeze_after_assign=2.0,
        max_images=200,
        max_days=3,
        log_func=None
    ):
        """
        save_dir            : Thư mục lưu ảnh khuôn mặt lạ
        min_confidence      : Ngưỡng confidence tối thiểu
        min_face_size       : Kích thước khuôn mặt nhỏ nhất (pixel)
        save_interval       : Khoảng cách tối thiểu giữa 2 lần lưu ảnh
        group_interval      : Khoảng thời gian để gom nhóm ảnh
        freeze_after_assign : Thời gian tạm dừng lưu sau khi assign
        max_images          : Số ảnh khuôn mặt lạ tối đa được giữ
        max_days            : Số ngày tối đa ảnh được tồn tại
        log_func            : Hàm ghi log (truyền từ web.py)
        """

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # ===== CONFIGURATION =====
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.save_interval = save_interval
        self.group_interval = group_interval
        self.freeze_after_assign = freeze_after_assign
        self.max_images = max_images
        self.max_days = max_days

        # ===== INTERNAL STATE =====
        self._last_save_time = 0.0
        self._last_assign_time = 0.0

        self.log_func = log_func

    # =================================================
    # LOG HELPER
    # =================================================
    def _log(self, message):
        if self.log_func:
            self.log_func(message)

    # =================================================
    # MAIN FUNCTION: HANDLE UNKNOWN FACE
    # =================================================
    def handle_unknown_face(self, frame, bbox, confidence):
        """
        frame      : Frame gốc (BGR)
        bbox       : (x1, y1, x2, y2)
        confidence : Độ tin cậy của khuôn mặt
        """

        current_time = time.time()

        # Không lưu nếu đang trong thời gian khóa sau assign
        if current_time - self._last_assign_time < self.freeze_after_assign:
            return False

        # Kiểm tra confidence
        if confidence < self.min_confidence:
            return False

        x1, y1, x2, y2 = bbox

        # Kiểm tra bounding box hợp lệ
        if x2 <= x1 or y2 <= y1:
            return False

        # Kiểm tra kích thước khuôn mặt
        if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
            return False

        # Giới hạn tần suất lưu ảnh
        if current_time - self._last_save_time < self.save_interval:
            return False

        # Cắt ảnh khuôn mặt
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return False

        # Lưu ảnh khuôn mặt lạ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"unknown_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)

        success = cv2.imwrite(filepath, face_img)
        if not success:
            self._log(f"Failed to save UNKNOWN face: {filename}")
            return False

        self._last_save_time = current_time
        self._log(f"Auto-captured UNKNOWN face: {filename}")

        # Dọn dẹp ảnh sau khi lưu
        self.cleanup()

        return True

    # =================================================
    # CLEANUP FUNCTIONS
    # =================================================
    def cleanup(self):
        """Gọi dọn dẹp theo số lượng và thời gian"""
        self._cleanup_by_count()
        self._cleanup_by_age()

    def _cleanup_by_count(self):
        files = self._get_sorted_files()

        if len(files) <= self.max_images:
            return

        remove_count = len(files) - self.max_images
        for filepath, _ in files[:remove_count]:
            try:
                os.remove(filepath)
                self._log(f"Auto-delete UNKNOWN (count): {os.path.basename(filepath)}")
            except FileNotFoundError:
                self._log("Delete failed: file not found")
            except PermissionError:
                self._log("Delete failed: permission denied")
            except OSError as e:
                self._log(f"Delete failed: {e}")

    def _cleanup_by_age(self):
        if self.max_days <= 0:
            return

        now = time.time()
        max_age = self.max_days * 24 * 3600

        for filepath, mtime in self._get_sorted_files():
            if now - mtime > max_age:
                try:
                    os.remove(filepath)
                    self._log(f"Auto-delete UNKNOWN (age): {os.path.basename(filepath)}")
                except FileNotFoundError:
                    self._log("Delete failed: file not found")
                except PermissionError:
                    self._log("Delete failed: permission denied")
                except OSError as e:
                    self._log(f"Delete failed: {e}")

    # =================================================
    # GROUPING FOR GALLERY DISPLAY
    # =================================================
    def get_groups(self):
        """
        Gom nhóm ảnh khuôn mặt lạ theo thời gian
        Trả về danh sách group cho gallery
        """

        files = self._get_sorted_files()
        groups = []
        current_group = []

        for filepath, mtime in files:
            item = {
                "filename": os.path.basename(filepath),
                "url": f"/{self.save_dir}/{os.path.basename(filepath)}",
                "time": mtime
            }

            if not current_group:
                current_group.append(item)
            elif mtime - current_group[-1]["time"] <= self.group_interval:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]

        if current_group:
            groups.append(current_group)

        return [
            {
                "group_id": f"group_{i}",
                "count": len(group),
                "preview": group[-1]["url"],
                "files": group
            }
            for i, group in enumerate(groups)
        ]

    # =================================================
    # FREEZE AFTER ASSIGN IDENTITY
    # =================================================
    def freeze_after_identity_assign(self):
        """Khóa lưu ảnh trong một khoảng thời gian sau khi gán danh tính"""
        self._last_assign_time = time.time()

    # =================================================
    # INTERNAL UTILITIES
    # =================================================
    def _get_sorted_files(self):
        """Lấy danh sách file ảnh và sắp xếp từ cũ đến mới"""
        files = []
        try:
            for name in os.listdir(self.save_dir):
                if not name.endswith(".jpg"):
                    continue
                path = os.path.join(self.save_dir, name)
                files.append((path, os.path.getmtime(path)))
        except FileNotFoundError:
            return []

        files.sort(key=lambda x: x[1])
        return files
