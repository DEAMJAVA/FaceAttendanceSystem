import os
import cv2
import numpy as np

import config


class SFaceRecognizer:
    def __init__(self, model_path=None):
        model_path = model_path or config.SFACE_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SFace model not found at '{model_path}'. "
                "Run `python download_models.py` first, or this backend "
                "isn't available and the fallback LBPH recognizer will be used."
            )
        self.model = cv2.FaceRecognizerSF.create(model_path, "")
        self.names = []          # list[str], index -> name
        self.embeddings = None   # (N, 128) float32

    def _embed(self, aligned_bgr_face):
        feature = self.model.feature(aligned_bgr_face)
        return feature.flatten()

    def enroll(self, name, aligned_face_crops):
        for crop in aligned_face_crops:
            emb = self._embed(crop)
            self.names.append(name)
            if self.embeddings is None:
                self.embeddings = emb.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])

    def save(self, path=None):
        path = path or config.EMBEDDINGS_PATH
        np.savez(path, names=np.array(self.names), embeddings=self.embeddings)

    def load(self, path=None):
        path = path or config.EMBEDDINGS_PATH
        data = np.load(path, allow_pickle=True)
        self.names = list(data["names"])
        self.embeddings = data["embeddings"]

    def recognize(self, aligned_face_crop):
        if self.embeddings is None or len(self.names) == 0:
            return None, 0.0
        query = self._embed(aligned_face_crop).reshape(1, -1)
        best_name, best_score = None, -1.0
        for name, emb in zip(self.names, self.embeddings):
            score = self.model.match(
                query, emb.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score, best_name = score, name
        if best_score >= config.SFACE_MATCH_THRESHOLD:
            return best_name, best_score
        return None, best_score


class LBPHRecognizer:

    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}

    def train(self, gray_faces, labels, label_map):
        self.model.train(gray_faces, np.array(labels))
        self.label_map = label_map

    def save(self, model_path=None, label_map_path=None):
        self.model.save(model_path or config.LBPH_MODEL_PATH)
        np.save(label_map_path or config.LABEL_MAP_PATH, self.label_map)

    def load(self, model_path=None, label_map_path=None):
        self.model.read(model_path or config.LBPH_MODEL_PATH)
        self.label_map = np.load(
            label_map_path or config.LABEL_MAP_PATH, allow_pickle=True
        ).item()

    def recognize(self, gray_face_crop):
        label, confidence = self.model.predict(gray_face_crop)
        if confidence < config.LBPH_CONFIDENCE_THRESHOLD:
            return self.label_map.get(label, "Unknown"), confidence
        return None, confidence


def create_recognizer():
    try:
        recognizer = SFaceRecognizer()
        print("[face_recognizer] Using SFace DNN recognizer.")
        return recognizer
    except FileNotFoundError as e:
        print(f"[face_recognizer] {e}")
        print("[face_recognizer] Falling back to LBPH recognizer.")
        return LBPHRecognizer()


def has_saved_model(recognizer):
    if isinstance(recognizer, SFaceRecognizer):
        return os.path.exists(config.EMBEDDINGS_PATH)
    return os.path.exists(config.LBPH_MODEL_PATH) and os.path.exists(config.LABEL_MAP_PATH)