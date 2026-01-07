import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as T

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_FOLDER = "data/face_train"
PROCESSED_FOLDER = "data/face_processed"   # chứa ảnh đã cắt và chuẩn hóa bằng MTCNN
EMBEDDING_FILE = "face_embeddings.pt"

DEBUG_DIR = "data/face_debug"   # tùy chọn

# =========================
# CHECK DATASET
# =========================
if not os.path.exists(TRAIN_FOLDER):
    raise RuntimeError("Face train folder not found")

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# =========================
# MODELS
# =========================
mtcnn = MTCNN(
    image_size=160,
    margin=10,
    min_face_size=20
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

known_faces = {}

print("[INFO] Starting face embedding processing")
print(f"[INFO] Using device: {DEVICE}")

# =========================
# TRAIN LOOP (ENROLLMENT)
# =========================
for person_name in os.listdir(TRAIN_FOLDER):
    person_path = os.path.join(TRAIN_FOLDER, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    total_images = 0
    used_images = 0

    # Thư mục lưu ảnh đã cắt
    processed_person_dir = os.path.join(PROCESSED_FOLDER, person_name)
    os.makedirs(processed_person_dir, exist_ok=True)

    # Debug
    debug_person_dir = os.path.join(DEBUG_DIR, person_name)
    os.makedirs(debug_person_dir, exist_ok=True)

    for filename in os.listdir(person_path):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(person_path, filename)
        total_images += 1

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open image {img_path}: {e}")
            continue

        # =========================
        # FACE DETECT + ALIGN + CROP
        # =========================
        face = mtcnn(img)
        if face is None:
            print(f"[WARN] No face detected in {img_path}")
            continue

        # =========================
        # SAVE PROCESSED FACE IMAGE
        # =========================
        face_img = T.ToPILImage()(face)
        processed_img_path = os.path.join(processed_person_dir, filename)
        face_img.save(processed_img_path)

        # Debug: lưu ảnh gốc detect thành công
        debug_img_path = os.path.join(debug_person_dir, filename)
        img.save(debug_img_path)

        # =========================
        # Trích xuất đặc trưng khuôn mặt
        # =========================
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(DEVICE))[0]
            embedding = embedding / embedding.norm()  # chuẩn hóa L2 cho vector để độ dài bằng 1
            embeddings.append(embedding)
            used_images += 1

    if len(embeddings) == 0:
        print(f"[WARN] No valid face data for {person_name}, skipped")
        continue

    # =========================
    # Tính trung bình các vector để lấy 1 vector đại diện cho khuôn mặt người
    # =========================
    mean_embedding = torch.stack(embeddings).mean(0)
    mean_embedding = mean_embedding / mean_embedding.norm()
    known_faces[person_name] = mean_embedding

    print(f"[INFO] {person_name}: {used_images}/{total_images} images used")

# =========================
# SAVE EMBEDDINGS
# =========================
torch.save(known_faces, EMBEDDING_FILE)

print(f"[DONE] Saved {len(known_faces)} identities to {EMBEDDING_FILE}")
print("[INFO] Processed face images saved to:", PROCESSED_FOLDER)
print("[INFO] Debug images saved to:", DEBUG_DIR)
