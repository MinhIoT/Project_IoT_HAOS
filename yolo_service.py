# =====================================================
# PERSON DETECTION + TRACKING ONLY
# =====================================================

import cv2                              # Thư viện xử lý ảnh
import time                             # Quản lý thời gian
import random                           # Sinh màu ngẫu nhiên
from ultralytics import YOLO            # YOLO (Ultralytics)

# =====================================================
# CLASS YOLOService
# =====================================================
class YOLOService:

    # -------------------------------------------------
    # KHỞI TẠO SERVICE
    # -------------------------------------------------
    def __init__(
        self,
        model_path,                     # Đường dẫn model YOLO
        conf_threshold=0.6,             # Ngưỡng confidence để lọc đối tượng
        iou_threshold=0.5,              # Ngưỡng IoU loại box trùng
        persist=True                    # Giữ trạng thái tracker id
    ):
        self.model = YOLO(model_path)   # Load model YOLO
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.persist = persist

        self.id_colors = {}             # LƯU MÀU THEO TRACK_ID
        self.last_reset = time.time()   # Thời điểm reset tracker

    # -------------------------------------------------
    # RESET TRACKER (KHI AI OFF / RTSP RECONNECT)
    # -------------------------------------------------
    def reset(self):
        self.model.tracker = None       # Xóa trạng thái tracker
        self.id_colors.clear()          # Xóa màu cũ
        self.last_reset = time.time()   # Cập nhật thời gian reset

    # -------------------------------------------------
    # LẤY MÀU THEO TRACK_ID (MỖI ID 1 MÀU)
    # -------------------------------------------------
    def _get_color(self, track_id):

        if track_id not in self.id_colors:
            self.id_colors[track_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )

        return self.id_colors[track_id]

    # -------------------------------------------------
    # TÍNH IOU GIỮA 2 BOUNDING BOX
    # -------------------------------------------------
    def _iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)    # chiều rộng
        interH = max(0, yB - yA)    # chiều cao
        interArea = interW * interH

        if interArea == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / (areaA + areaB - interArea)
        # IOU = (diện tích giao nhau) / (diện tích hợp)
    # -------------------------------------------------
    # LOẠI BỎ BOUNDING BOX TRÙNG
    # -------------------------------------------------
    def _remove_overlap(self, persons):

        persons = sorted(
            persons,
            key=lambda x: x["confidence"], # sắp xếp người theo độ tin cậy cao nhất
            reverse=True
        )

        filtered = []

        for p in persons:
            keep = True
            for f in filtered:
                if self._iou(p["bbox"], f["bbox"]) > self.iou_threshold: # nếu IoU bị lớn hơn ngưỡng -> box bị trùng
                    keep = False
                    break
            if keep:
                filtered.append(p)

        return filtered

    # -------------------------------------------------
    # PHÁT HIỆN + THEO DÕI NGƯỜI
    # -------------------------------------------------
    def detect(self, frame, pad=10):

        results = self.model.track(
            frame,
            classes=[0],                # chỉ phát hiện người
            conf=self.conf_threshold,
            persist=self.persist,       # giữ nguyên ID của từng người
            verbose=False
        )

        result = results[0]
        persons = []

        if result.boxes is None or result.boxes.id is None:
            return persons
        # Duyệt từng người được phát hiện
        for box, conf, tid in zip(
            result.boxes.xyxy,     # tọa độ bounding box
            result.boxes.conf,     # độ tin cậy
            result.boxes.id        # id
        ):
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1 - pad)        # thêm padding
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            persons.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "track_id": int(tid)
            })

        return self._remove_overlap(persons)

    # -------------------------------------------------
    # VẼ BOUNDING BOX (MỖI ID 1 MÀU)
    # -------------------------------------------------
    def draw(self, frame, persons):

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            tid = p["track_id"]
            conf = p["confidence"]

            color = self._get_color(tid)     # LẤY MÀU THEO ID
            label = f"ID {tid} {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        return frame
