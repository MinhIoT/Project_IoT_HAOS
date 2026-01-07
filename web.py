# =====================================================
# RTSP -> OpenCV -> MJPEG
# YOLO + Face Recognition
# Add new person "unknow" for detect
# MQTT publish to Home Assistant
# =====================================================
import os
import cv2
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, Response, render_template, jsonify, request
from PIL import Image, UnidentifiedImageError

from yolo_service import YOLOService
from face_service import FaceService
from mqtt_haos import MQTTClientHAOS
from QuanLyKhuonMatLa import UnknownFaceManager

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

RTSP_URL = os.getenv("RTSP_URL")
FLASK_URL = os.getenv("FLASK_URL")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
FACE_EMB_FILE = os.getenv("FACE_EMB_FILE", "face_embeddings.pt")

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

MQTT_TOPIC_YOLO = "home/camera/yolo"
MQTT_TOPIC_FACE = "home/camera/face"

# =====================================================
# CONFIG
# =====================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
JPEG_QUALITY = 80

CAPTURE_DIR = "static/captures"
YOLO_DIR = "static/yolo_captures"
FACE_DIR = "static/face_captures"
UNKNOWN_DIR = "static/unknown_captures"

UNKNOWN_MIN_CONF = 0.5
UNKNOWN_MIN_SIZE = 80
UNKNOWN_SAVE_INTERVAL = 3.0
UNKNOWN_GROUP_INTERVAL = 5.0
ASSIGN_FREEZE_SEC = 2.0

MQTT_INTERVAL = 2.0

for d in (CAPTURE_DIR, YOLO_DIR, FACE_DIR, UNKNOWN_DIR):
    os.makedirs(d, exist_ok=True)

# =====================================================
# FLASK
# =====================================================
app = Flask(__name__)
app.secret_key = "ai-camera-gateway"

# =====================================================
# Frame mới nhất từ Camera
# =====================================================
latest_frame_raw = None    # frame gốc
latest_frame_draw = None   # frame đã vẽ bounding box
latest_frame_ts = 0.0      # thời gian đọc frame gần nhất đọc được từ Camera
# Trạng thái hệ thống
camera_running = True
AI_MODE = False

frame_lock = threading.Lock()

# =====================================================
# LOG SYSTEM
# =====================================================
SYSTEM_LOGS = []
CAPTURE_LOGS = []
LOG_LIMIT = 50


def _push_log(buf, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    buf.append(f"[{ts}] {msg}")
    if len(buf) > LOG_LIMIT:
        buf.pop(0)


def push_system_log(msg):
    _push_log(SYSTEM_LOGS, msg)


def push_capture_log(msg):
    _push_log(CAPTURE_LOGS, msg)


# =====================================================
# UTILS
# =====================================================
def next_index(folder):
    nums = []
    for f in os.listdir(folder):
        if not f.endswith(".jpg"):
            continue
        base = os.path.splitext(f)[0]
        try:
            nums.append(int(base.split("_", 1)[0]))
        except ValueError:
            continue
    return max(nums, default=0) + 1

# =====================================================
# INIT AI SERVICES
# =====================================================
yolo = YOLOService(
    model_path=YOLO_MODEL_PATH,
    conf_threshold=0.6,
    iou_threshold=0.5,
    persist=True
)

face_service = FaceService(
    embeddings_file=FACE_EMB_FILE,
    match_threshold=0.6,
    min_face_prob=0.9
)
# INIT UNKNOWN FACE MANAGER

unknown_manager = UnknownFaceManager(
    save_dir=UNKNOWN_DIR,
    min_confidence=UNKNOWN_MIN_CONF,
    min_face_size=UNKNOWN_MIN_SIZE,
    save_interval=UNKNOWN_SAVE_INTERVAL,
    group_interval=UNKNOWN_GROUP_INTERVAL,
    freeze_after_assign=ASSIGN_FREEZE_SEC,
    max_images=200,
    max_days=3,
    log_func=push_system_log
)

# =====================================================
# INIT MQTT
# =====================================================
mqtt = MQTTClientHAOS(
    broker=MQTT_BROKER,
    port=MQTT_PORT,
    username=MQTT_USERNAME,
    password=MQTT_PASSWORD,
    client_id="ai_camera_gateway"
)
mqtt.connect()

# =====================================================
# CAMERA THREAD -> đọc frame từ Camera
# Không xử lý AI chỉ cập nhật frame mới nhất
# =====================================================
def camera_reader():
    global latest_frame_raw, latest_frame_ts

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        push_system_log("ERROR: Cannot open RTSP")
        return

    push_system_log("RTSP connected")

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        with frame_lock:
            latest_frame_raw = frame
            latest_frame_ts = time.time()

    cap.release()

# =====================================================
# AI THREAD
# =====================================================
def ai_worker():
    global latest_frame_draw

    last_ts = 0.0
    last_mqtt_yolo = 0.0
    last_mqtt_face = 0.0

    while camera_running:
        if not AI_MODE:
            time.sleep(0.05)
            continue

        with frame_lock:                # lấy frame mới nhất từ Camera
            frame = latest_frame_raw
            ts = latest_frame_ts

        if frame is None or ts <= last_ts:
            time.sleep(0.01)
            continue

        last_ts = ts
        draw = frame.copy()
        clean = frame.copy()
        # YOLO phát hiện người
        persons = yolo.detect(draw)
        draw = yolo.draw(draw, persons)

        now = time.time()
        if persons and now - last_mqtt_yolo > MQTT_INTERVAL:
            mqtt.publish(
                MQTT_TOPIC_YOLO,
                {
                    "type": "realtime",
                    "detections": [
                        {
                            "class_name": "person", "confidence": round(p["confidence"], 2)
                        }
                        for p in persons
                    ],
                    "time": datetime.now().isoformat()
                }
            )
            last_mqtt_yolo = now
        # Phát hiện khuôn mặt người trong vùng ảnh quan tâm từ YOLO
        frame_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        face_events = []

        for p in persons:
            faces = face_service.recognize_in_roi(frame_rgb, p["bbox"])
            for f in faces:
                x1, y1, x2, y2 = f["bbox"]
                name = f["name"]
                conf = f["confidence"]

                if name == "Unknown":
                    unknown_manager.handle_unknown_face(clean, (x1, y1, x2, y2), conf)

                face_events.append({
                    "name": name,
                    "confidence": round(conf, 2)
                })

                color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    draw, f"{name} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )

        if face_events and now -last_mqtt_face > MQTT_INTERVAL:
            mqtt.publish(
                MQTT_TOPIC_FACE,
                {
                    "type": "realtime",
                    "faces": face_events,
                    "time": datetime.now().isoformat()
                }
            )
            last_mqtt_face = now

        with frame_lock:
            latest_frame_draw = draw

        time.sleep(0.03)

# =====================================================
# MJPEG STREAM
# =====================================================
def mjpeg_generator():
    while camera_running:
        with frame_lock:
            frame = latest_frame_draw if AI_MODE else latest_frame_raw

        if frame is None:
            time.sleep(0.01)
            continue

        ret, jpg = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() + b"\r\n"
        )
        time.sleep(0.03)

# =====================================================
# web flask
# =====================================================
@app.route("/")
def index():                               # giao diện web
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/unknown_faces")
def unknown_faces():
    return jsonify({
        "groups": unknown_manager.get_groups()
    })

@app.route("/assign_face", methods=["POST"])
def assign_face():
    data = request.json or {}
    name = data.get("name", "")
    files = data.get("files", [])

    count = 0
    for f in files:
        path = os.path.join(UNKNOWN_DIR, f)
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            continue

        if face_service.add_face(name, img):
            count += 1

        try:
            os.remove(path)
        except (FileNotFoundError, PermissionError):
            pass

    unknown_manager.freeze_after_identity_assign()

    mqtt.publish(
        MQTT_TOPIC_FACE,
        {
            "event": "assign_identity",
            "name": name,
            "count": count,
            "time": datetime.now().isoformat()
        }
    )

    return jsonify({"status": "ok", "count": count})

@app.route("/capture_normal")
def capture_normal():
    with frame_lock:
        if latest_frame_raw is None:
            return jsonify({"error": "No frame"}), 400
        frame = latest_frame_raw.copy()

    idx = next_index(CAPTURE_DIR)
    filename = f"{idx:04d}_normal.jpg"
    cv2.imwrite(os.path.join(CAPTURE_DIR, filename), frame)
    push_capture_log(f"Captured NORMAL: {filename}")

    image_url = f"/{CAPTURE_DIR}/{filename}"

    mqtt.publish(
        MQTT_TOPIC_YOLO,
        {
            "type": "capture_normal",
            "image_url": image_url,
            "time": datetime.now().isoformat()
        }
    )

    return jsonify({"image_url": image_url})


@app.route("/capture_yolo")
def capture_yolo():
    with frame_lock:
        if latest_frame_raw is None:
            return jsonify({"error": "No frame"}), 400
        frame = latest_frame_raw.copy()

    persons = yolo.detect(frame)
    frame = yolo.draw(frame, persons)

    idx = next_index(YOLO_DIR)
    filename = f"{idx:04d}_yolo.jpg"
    cv2.imwrite(os.path.join(YOLO_DIR, filename), frame)
    push_capture_log(f"Captured YOLO: {filename}")

    relative_url = f"/static/yolo_captures/{filename}"
    absolute_url = f"{FLASK_URL}{relative_url}"

    detections = [
        {"class_name": "person", "confidence": round(p["confidence"], 2)}
        for p in persons
    ]

    mqtt.publish(
        MQTT_TOPIC_YOLO,
        {
            "type": "capture_yolo",
            "image_url": absolute_url,
            "detections": detections,
            "time": datetime.now().isoformat()
        }
    )

    return jsonify({"image_url": relative_url})


@app.route("/capture_face")
def capture_face():
    with frame_lock:
        if latest_frame_raw is None:
            return jsonify({"error": "No frame"}), 400
        frame = latest_frame_raw.copy()

    # BGR -> RGB cho FaceNet / MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    faces = face_service.recognize(frame_rgb)

    if not faces:
        push_capture_log("Capture FACE failed: no face")
        return jsonify({"error": "No face detected"}), 400

    saved = []
    idx = next_index(FACE_DIR)

    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        name = f["name"]
        conf = f["confidence"]

        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)

        # Vẽ bounding box khuôn mặt
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        filename = f"{idx:04d}_{name}.jpg"
        cv2.imwrite(os.path.join(FACE_DIR, filename), frame)
        saved.append(filename)
        idx += 1

    for filename in saved:
        push_capture_log(f"Captured FACE: {filename}")

    relative_url = f"/static/face_captures/{saved[-1]}"
    absolute_url = f"{FLASK_URL}{relative_url}"

    mqtt.publish(
        MQTT_TOPIC_FACE,
        {
            "type": "capture_face",
            "image_url": absolute_url,
            "faces": [
                {
                    "name": f.split("_", 1)[1].replace(".jpg", "")
                }
                for f in saved
            ],
            "time": datetime.now().isoformat()
        }
    )

    return jsonify({
        "image_url": relative_url,
        "faces": [
            {"name": f.split("_", 1)[1].replace(".jpg", "")}
            for f in saved
        ]
    })

@app.route("/ai_on")
def ai_on():
    global AI_MODE
    AI_MODE = True
    yolo.reset()
    return jsonify({"ai_mode": True})


@app.route("/ai_off")
def ai_off():
    global AI_MODE
    AI_MODE = False
    return jsonify({"ai_mode": False})

@app.route("/ai_status")
def ai_status():
    return jsonify({
        "ai_mode": AI_MODE,
        "system_logs": SYSTEM_LOGS[-10:],
        "capture_logs": CAPTURE_LOGS[-10:]
    })

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    threading.Thread(target=camera_reader, daemon=True).start()
    threading.Thread(target=ai_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
