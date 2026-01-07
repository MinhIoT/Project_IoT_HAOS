# =====================================================
# face_service.py
# FACE RECOGNITION WITH MTCNN + FACENET
# =====================================================

import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

class FaceService:
    def __init__(
        self,
        embeddings_file="face_embeddings.pt",
        device=None,
        match_threshold=0.75,
        min_face_prob=0.9
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.embeddings_file = embeddings_file
        self.MATCH_THRESHOLD = match_threshold
        self.MIN_FACE_PROB = min_face_prob

        # name -> embedding (centroid)
        self.known_faces: dict[str, torch.Tensor] = {}

        self._load_embeddings()

        self.mtcnn = MTCNN(
            image_size=160,
            margin=10,
            min_face_size=20,
            device=self.device
        )

        self.resnet = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )

        print(f"[FaceService] Device: {self.device}")
        print(f"[FaceService] Loaded {len(self.known_faces)} identities")

    # =====================================================
    # LOAD / SAVE
    # =====================================================
    def _load_embeddings(self):
        if not os.path.exists(self.embeddings_file):
            self.known_faces = {}
            return

        data = torch.load(self.embeddings_file, map_location=self.device)

        self.known_faces = {
            name: emb.to(self.device) / emb.norm()
            for name, emb in data.items()
        }

    def save_embeddings(self):
        torch.save(self.known_faces, self.embeddings_file)

    # =====================================================
    # Trích xuất đặc trưng khuôn mặt
    # =====================================================
    def _extract_embedding(self, face_img):
        if isinstance(face_img, Image.Image):
            img = face_img.convert("RGB")
        else:
            img = Image.fromarray(face_img).convert("RGB")

        face_tensor = self.mtcnn(img)
        if face_tensor is None:
            return None

        with torch.no_grad():
            emb = self.resnet(
                face_tensor.unsqueeze(0).to(self.device)
            )[0]
            emb = emb / emb.norm()

        return emb

    # =====================================================
    # So khớp khuôn mặt dựa vào cosine similarity
    # =====================================================
    def _match(self, emb):
        best_name = "Unknown"
        best_score = 0.0

        for name, db_emb in self.known_faces.items():
            score = torch.cosine_similarity(
                emb, db_emb, dim=0
            ).item()

            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.MATCH_THRESHOLD:
            return best_name, best_score

        return "Unknown", best_score

    # =====================================================
    # RECOGNIZE IN ROI
    # =====================================================
    def recognize_in_roi(self, frame_rgb, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame_rgb[y1:y2, x1:x2]

        if roi.size == 0:
            return []

        results = []

        boxes, probs = self.mtcnn.detect(roi)
        if boxes is None:
            return results

        for box, prob in zip(boxes, probs):
            if prob < self.MIN_FACE_PROB:
                continue

            fx1, fy1, fx2, fy2 = box.astype(int)
            face_crop = roi[fy1:fy2, fx1:fx2]

            if face_crop.size == 0:
                continue

            emb = self._extract_embedding(face_crop)
            if emb is None:
                continue

            name, score = self._match(emb)

            results.append({
                "name": name,
                "confidence": float(score),
                "bbox": [
                    x1 + fx1,
                    y1 + fy1,
                    x1 + fx2,
                    y1 + fy2
                ],
                "face_crop": face_crop
            })

        return results

    def recognize(self, frame_rgb):
        """
        Detect & recognize faces on full frame.
        Wrapper for recognize_in_roi()
        """
        if frame_rgb is None:
            return []

        h, w, _ = frame_rgb.shape
        full_bbox = (0, 0, w, h)

        return self.recognize_in_roi(frame_rgb, full_bbox)
    # =====================================================
    # ADD FACE (CENTROID UPDATE)
    # =====================================================
    def add_face(self, name, face_img):
        emb = self._extract_embedding(face_img)
        if emb is None:
            return False

        if name in self.known_faces:
            # ---- Cập nhật lại vector trung bình đặc trưng  ----
            old = self.known_faces[name]
            new = (old + emb) / 2
            self.known_faces[name] = new / new.norm()
        else:
            self.known_faces[name] = emb

        self.save_embeddings()
        print(f"[FaceService] Updated face: {name}")
        return True

    # =====================================================
    # LIST PEOPLE
    # =====================================================
    def list_people(self):
        return list(self.known_faces.keys())
