import os
from collections import Counter

import cv2
import numpy as np

import config


def _cosine_sim(query, matrix):
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return (m @ q.T).flatten()


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
        self.names = []
        self.centroids = None
        self.raw_names = []
        self.raw_embeddings = None

    def align(self, frame, detection):
        if detection.landmarks is not None:
            row = detection.as_yunet_row().reshape(1, -1)
            return self.model.alignCrop(frame, row)
        x, y, w, h = detection.box
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, config.FACE_ALIGN_SIZE)

    def _embed(self, aligned_bgr_face):
        feature = self.model.feature(aligned_bgr_face)
        return feature.flatten()

    def enroll(self, name, aligned_face_crops):
        embs = np.array(
            [self._embed(c) for c in aligned_face_crops], dtype=np.float32
        )
        if len(embs) == 0:
            return

        if len(embs) > 3:
            provisional_centroid = embs.mean(axis=0, keepdims=True)
            sims = _cosine_sim(provisional_centroid, embs)
            keep = sims >= (sims.mean() - 1.5 * sims.std())
            if keep.any():
                embs = embs[keep]

        self.raw_names.extend([name] * len(embs))
        self.raw_embeddings = (
            embs if self.raw_embeddings is None
            else np.vstack([self.raw_embeddings, embs])
        )

        centroid = embs.mean(axis=0)
        if name in self.names:
            idx = self.names.index(name)
            self.centroids[idx] = centroid
        else:
            self.names.append(name)
            self.centroids = (
                centroid.reshape(1, -1) if self.centroids is None
                else np.vstack([self.centroids, centroid.reshape(1, -1)])
            )

    def save(self, path=None):
        path = path or config.EMBEDDINGS_PATH
        np.savez(
            path,
            names=np.array(self.names),
            centroids=self.centroids,
            raw_names=np.array(self.raw_names),
            raw_embeddings=self.raw_embeddings,
        )

    def load(self, path=None):
        path = path or config.EMBEDDINGS_PATH
        data = np.load(path, allow_pickle=True)
        self.names = list(data["names"])
        self.centroids = data["centroids"]
        self.raw_names = list(data["raw_names"])
        self.raw_embeddings = data["raw_embeddings"]

    def recognize(self, aligned_face_crop):
        if self.centroids is None or len(self.names) == 0:
            return None, 0.0

        query = self._embed(aligned_face_crop).reshape(1, -1)
        scores = _cosine_sim(query, self.centroids)
        order = np.argsort(scores)[::-1]
        best_idx = order[0]
        best_score = scores[best_idx]

        if best_score < config.SFACE_MATCH_THRESHOLD:
            return None, best_score

        if len(self.names) > 1:
            second_score = scores[order[1]]
            if (best_score - second_score) < config.SFACE_MATCH_MARGIN:
                return None, best_score  # too close to call

        if self.raw_embeddings is not None and len(self.raw_embeddings) > 0:
            raw_scores = _cosine_sim(query, self.raw_embeddings)
            k = min(config.SFACE_KNN_K, len(raw_scores))
            top_k_idx = np.argsort(raw_scores)[::-1][:k]
            top_k_names = [self.raw_names[i] for i in top_k_idx]
            vote_name, vote_count = Counter(top_k_names).most_common(1)[0]
            if (
                vote_name != self.names[best_idx]
                or vote_count / k < config.SFACE_KNN_AGREEMENT
            ):
                return None, best_score

        return self.names[best_idx], best_score


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