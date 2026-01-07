import cv2
import os
import time
import shutil
from dotenv import load_dotenv

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
SAVE_ROOT = "data/face_train"

SAVE_INTERVAL = 0.2
MAX_IMAGES = 100
RECONNECT_DELAY = 1.0

# =====================================================
# HAAR CASCADE CONFIG
# =====================================================
CASCADE_PATH = "C:/Users/tvmin/PycharmProjects/DATN_Final/models/haarcascade_frontalface_default.xml"

SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 6
MIN_FACE_SIZE = (80, 80)

# =====================================================
# Xác thực mô hình
# =====================================================
if RTSP_URL is None:
    raise RuntimeError("RTSP_URL not found in .env file")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("Cannot load Haar Cascade model")

os.makedirs(SAVE_ROOT, exist_ok=True)

# =====================================================
# RTSP OPEN (SAFE)
# =====================================================
def open_rtsp():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# =====================================================
# ASK PERSON NAME
# =====================================================
def ask_person_name():
    while True:
        name = input("Nhập tên người cần lưu khuôn mặt: ").strip()
        if name:
            return name
        print("Tên không hợp lệ, vui lòng nhập lại")

# =====================================================
# HANDLE EXISTING DATA
# =====================================================
def handle_existing_folder(name):
    person_dir = os.path.join(SAVE_ROOT, name)

    if not os.path.exists(person_dir):
        os.makedirs(person_dir, exist_ok=True)
        return person_dir

    print(f"\nĐã tồn tại dữ liệu cho người '{name}'")
    print("1 - Dùng tiếp (append ảnh mới)")
    print("2 - Ghi đè (xóa toàn bộ dữ liệu cũ)")
    print("3 - Hủy")

    choice = input("Nhập lựa chọn (1/2/3): ").strip()

    if choice == "1":
        return person_dir

    if choice == "2":
        shutil.rmtree(person_dir)
        os.makedirs(person_dir, exist_ok=True)
        print(f"[INFO] Đã xóa dữ liệu cũ của {name}")
        return person_dir

    print("[INFO] Hủy thao tác")
    return None

# =====================================================
# COLLECT IMAGES (NO CROP)
# =====================================================
def collect_images(person_name):
    save_dir = handle_existing_folder(person_name)
    if save_dir is None:
        return

    cap = open_rtsp()
    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream")
        return

    count = len(os.listdir(save_dir))
    last_save = 0
    paused = False

    print(
        "\n[INFO] Thu thập ảnh FULL FRAME bằng Haar Cascade\n"
        "      P : Pause / Resume\n"
        "      Q : Quit\n"
    )

    while True:
        ret, frame = cap.read()

        # ===== HANDLE RTSP DROP =====
        if not ret or frame is None:
            print("[WARN] RTSP mất kết nối – reconnect...")
            cap.release()
            time.sleep(RECONNECT_DELAY)
            cap = open_rtsp()
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE
        )

        # Vẽ bounding box cho tất cả khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ===== SAVE FULL FRAME IF FACE DETECTED =====
        now = time.time()
        if (
            not paused
            and len(faces) > 0          # 👈 chỉ cần có khuôn mặt
            and now - last_save >= SAVE_INTERVAL
            and count < MAX_IMAGES
        ):
            filename = f"{person_name}_{count + 1:03d}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), frame)

            count += 1
            last_save = now
            print(f"[SAVE] {filename}")

        status = "PAUSED" if paused else "RUNNING"
        cv2.putText(
            frame,
            f"{status} | Collected {count}/{MAX_IMAGES}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if paused else (255, 255, 0),
            2
        )

        cv2.imshow("Collect Face - Haar Cascade (Full Frame)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('p'):
            paused = not paused
            print("[INFO] Pause" if paused else "[INFO] Resume")

        if count >= MAX_IMAGES:
            print("[INFO] Đã đủ số lượng ảnh")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Hoàn tất thu thập dữ liệu")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    name = ask_person_name()
    collect_images(name)
