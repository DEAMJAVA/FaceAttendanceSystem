import os
import cv2
import numpy as np

import config


class Detection:

    __slots__ = ("x", "y", "w", "h", "confidence", "landmarks")

    def __init__(self, x, y, w, h, confidence=1.0, landmarks=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.confidence = confidence
        self.landmarks = landmarks

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    def as_yunet_row(self):
        if self.landmarks is None:
            raise ValueError(
                "This detection has no landmarks (came from the Haar "
                "fallback detector), so it can't be used with SFace's "
                "alignCrop. Naive-crop alignment is used instead in that case."
            )
        row = np.zeros((15,), dtype=np.float32)
        row[0:4] = [self.x, self.y, self.w, self.h]
        row[4:14] = self.landmarks.astype(np.float32).flatten()
        row[14] = self.confidence
        return row


class HaarCascadeDetector:

    def __init__(self):
        cascade_path = self._resolve_cascade_path()
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from '{cascade_path}' "
                "(file found but OpenCV could not parse it)."
            )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _resolve_cascade_path():
        for path in config.candidate_haarcascade_paths():
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Could not locate haarcascade_frontalface_default.xml. "
            "Checked: " + ", ".join(config.candidate_haarcascade_paths()) + ". "
            "Make sure opencv-python or opencv-contrib-python is installed "
            "correctly (`pip install --force-reinstall opencv-contrib-python`)."
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.HAAR_SCALE_FACTOR,
            minNeighbors=config.HAAR_MIN_NEIGHBORS,
            minSize=config.HAAR_MIN_SIZE,
        )
        return [Detection(x, y, w, h, confidence=1.0) for (x, y, w, h) in boxes]


class YuNetDetector:

    def __init__(self, model_path=None, input_size=(320, 320)):
        model_path = model_path or config.YUNET_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at '{model_path}'. "
                "Run `python download_models.py` first, or this backend "
                "isn't available and the fallback Haar detector will be used."
            )
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_threshold=config.YUNET_SCORE_THRESHOLD,
            nms_threshold=config.YUNET_NMS_THRESHOLD,
            top_k=config.YUNET_TOP_K,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        results = []
        if faces is None:
            return results
        for f in faces:
            x, y, bw, bh = f[0:4].astype(int)
            x, y = max(0, x), max(0, y)
            bw, bh = max(0, bw), max(0, bh)
            confidence = float(f[14])
            landmarks = f[4:14].reshape(5, 2)
            if bw < config.MIN_FACE_SIZE or bh < config.MIN_FACE_SIZE:
                continue
            results.append(Detection(x, y, bw, bh, confidence, landmarks))
        return results


def create_detector():
    try:
        detector = YuNetDetector()
        print("[face_detector] Using YuNet DNN detector.")
        return detector
    except FileNotFoundError as e:
        print(f"[face_detector] {e}")
        print("[face_detector] Falling back to Haar cascade detector.")
        return HaarCascadeDetector()


def align_face(frame, detection, output_size=config.FACE_ALIGN_SIZE):
    x, y, w, h = detection.box
    if detection.landmarks is not None:
        right_eye, left_eye = detection.landmarks[0], detection.landmarks[1]
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
        crop = rotated[max(0, y):y + h, max(0, x):x + w]
    else:
        crop = frame[max(0, y):y + h, max(0, x):x + w]

    if crop.size == 0:
        return None
    return cv2.resize(crop, output_size)