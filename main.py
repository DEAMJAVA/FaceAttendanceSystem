import argparse
from collections import Counter, deque

import cv2

import config
from face_detector import create_detector, align_face
from face_recognizer import create_recognizer, has_saved_model, SFaceRecognizer
from attendance import AttendanceLog
from dataset import capture_faces
from train import train_model


class FaceTracker:

    def __init__(self, max_center_dist=None, history=None, confirm_ratio=None):
        self.max_center_dist = max_center_dist or config.TRACK_MAX_CENTER_DIST
        self.history = history or config.TRACK_HISTORY
        self.confirm_ratio = confirm_ratio or config.TRACK_CONFIRM_RATIO
        self._tracks = {}  # id -> {"center": (x, y), "votes": deque[str|None]}
        self._next_id = 0

    def update(self, det, name):
        cx, cy = det.x + det.w / 2, det.y + det.h / 2
        best_id, best_dist = None, self.max_center_dist
        for tid, t in self._tracks.items():
            dist = ((t["center"][0] - cx) ** 2 + (t["center"][1] - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_id = dist, tid

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self._tracks[best_id] = {
                "center": (cx, cy),
                "votes": deque(maxlen=self.history),
            }

        self._tracks[best_id]["center"] = (cx, cy)
        self._tracks[best_id]["votes"].append(name)
        return best_id

    def confirmed_name(self, track_id):
        votes = self._tracks[track_id]["votes"]
        if len(votes) < self.history:
            return None
        winner, count = Counter(votes).most_common(1)[0]
        if winner is not None and count / len(votes) >= self.confirm_ratio:
            return winner
        return None


def run_recognition(camera_index=None):
    camera_index = config.CAMERA_INDEX if camera_index is None else camera_index
    detector = create_detector()
    recognizer = create_recognizer()

    if not has_saved_model(recognizer):
        print("[main] No trained model found. Run `python main.py --train` first.")
        return
    recognizer.load()

    attendance = AttendanceLog()
    tracker = FaceTracker()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {camera_index}).")

    print("[main] Starting recognition. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for det in detector.detect(frame):
                if isinstance(recognizer, SFaceRecognizer):
                    aligned = recognizer.align(frame, det)
                    if aligned is None:
                        continue
                    name, score = recognizer.recognize(aligned)
                else:
                    aligned = align_face(frame, det)
                    if aligned is None:
                        continue
                    gray_crop = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    name, score = recognizer.recognize(gray_crop)

                track_id = tracker.update(det, name)
                confirmed = tracker.confirmed_name(track_id)
                if confirmed:
                    attendance.mark(confirmed)

                x, y, w, h = det.box
                if name:
                    color = (0, 255, 0) if confirmed else (0, 165, 255)
                    label = f"{name} ({score:.2f})" if confirmed else f"{name}?"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("AI Attendance System (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_capture(camera_index=None):
    name = input("Enter the person's name: ").strip().capitalize()
    if not name:
        print("[main] Name cannot be empty.")
        return
    detector = create_detector()
    recognizer = create_recognizer()
    capture_faces(name, detector, camera_index=camera_index, recognizer=recognizer)


def main():
    parser = argparse.ArgumentParser(description="AI Attendance System")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--capture", action="store_true", help="capture a new person's face images")
    group.add_argument("-t", "--train", action="store_true", help="(re)train the recognition model")
    parser.add_argument(
        "--camera", type=int, default=None,
        help=f"camera index to use (overrides config.CAMERA_INDEX, default {config.CAMERA_INDEX})",
    )
    args = parser.parse_args()

    if args.capture:
        run_capture(camera_index=args.camera)
    elif args.train:
        train_model()
    else:
        run_recognition(camera_index=args.camera)


if __name__ == "__main__":
    main()