import os
import cv2
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = getattr(sys, "_MEIPASS", BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = BASE_DIR

IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendances")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


YUNET_MODEL_PATH = os.path.join(BUNDLE_DIR, "models", "face_detection_yunet.onnx")
SFACE_MODEL_PATH = os.path.join(BUNDLE_DIR, "models", "face_recognition_sface.onnx")
LBPH_MODEL_PATH = os.path.join(MODELS_DIR, "trainer.yml")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.npy")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npz")


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def candidate_haarcascade_paths(filename="haarcascade_frontalface_default.xml"):
    candidates = []
    data_attr = getattr(cv2, "data", None)
    if data_attr is not None and hasattr(data_attr, "haarcascades"):
        candidates.append(os.path.join(cv2.data.haarcascades, filename))
    cv2_pkg_dir = os.path.dirname(cv2.__file__)
    candidates.append(os.path.join(cv2_pkg_dir, "data", filename))
    candidates.append(os.path.join("/usr/share/opencv4/haarcascades", filename))
    candidates.append(os.path.join("/usr/local/share/opencv4/haarcascades", filename))
    return candidates

YUNET_SCORE_THRESHOLD = 0.8
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 10
MIN_FACE_SIZE = 40

HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 6
HAAR_MIN_SIZE = (MIN_FACE_SIZE, MIN_FACE_SIZE)

SFACE_MATCH_THRESHOLD = 0.363
SFACE_MATCH_MARGIN = 0.05
SFACE_KNN_K = 5
SFACE_KNN_AGREEMENT = 0.6

LBPH_CONFIDENCE_THRESHOLD = 70

CAPTURE_IMAGE_COUNT = 30
FACE_ALIGN_SIZE = (112, 112)
CAPTURE_BLUR_THRESHOLD = 60.0
CAPTURE_FRAME_INTERVAL = 3

TRACK_HISTORY = 6
TRACK_CONFIRM_RATIO = 0.6
TRACK_MAX_CENTER_DIST = 80

CAMERA_INDEX = 0