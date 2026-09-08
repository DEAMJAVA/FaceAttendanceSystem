import os
import cv2

import config
from face_recognizer import SFaceRecognizer, LBPHRecognizer, create_recognizer
from face_detector import create_detector, align_face


def train_model():
    recognizer = create_recognizer()
    detector = create_detector() if isinstance(recognizer, SFaceRecognizer) else None

    people = [
        d for d in os.listdir(config.IMAGES_DIR)
        if os.path.isdir(os.path.join(config.IMAGES_DIR, d))
    ]
    if not people:
        print(f"[train] No people found in '{config.IMAGES_DIR}'. Capture some faces first.")
        return

    if isinstance(recognizer, SFaceRecognizer):
        _train_sface(recognizer, detector, people)
    else:
        _train_lbph(recognizer, people)


def _train_sface(recognizer, detector, people):
    for name in people:
        person_dir = os.path.join(config.IMAGES_DIR, name)
        crops = []
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            if img.shape[:2] == config.FACE_ALIGN_SIZE[::-1]:
                crops.append(img)
            else:
                dets = detector.detect(img)
                if dets:
                    aligned = recognizer.align(img, dets[0])
                    if aligned is not None:
                        crops.append(aligned)
        if crops:
            recognizer.enroll(name, crops)
            print(f"[train] Enrolled {len(crops)} samples for '{name}'.")
        else:
            print(f"[train] No usable images for '{name}', skipped.")

    recognizer.save()
    print(f"[train] Saved embeddings to {config.EMBEDDINGS_PATH}")


def _train_lbph(recognizer, people):
    gray_faces, labels = [], []
    label_map = {}

    for current_id, name in enumerate(people):
        label_map[current_id] = name
        person_dir = os.path.join(config.IMAGES_DIR, name)
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            gray_faces.append(img)
            labels.append(current_id)

    if not gray_faces:
        print("[train] No usable images found.")
        return

    recognizer.train(gray_faces, labels, label_map)
    recognizer.save()
    print(f"[train] Saved LBPH model to {config.LBPH_MODEL_PATH}")


if __name__ == "__main__":
    train_model()