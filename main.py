import argparse
import cv2

import config
from face_detector import create_detector, align_face
from face_recognizer import create_recognizer, has_saved_model, SFaceRecognizer
from attendance import AttendanceLog
from dataset import capture_faces
from train import train_model


def run_recognition():
    detector = create_detector()
    recognizer = create_recognizer()

    if not has_saved_model(recognizer):
        print("[main] No trained model found. Run `python main.py --train` first.")
        return
    recognizer.load()

    attendance = AttendanceLog()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("[main] Starting recognition. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for det in detector.detect(frame):
                aligned = align_face(frame, det)
                if aligned is None:
                    continue

                if isinstance(recognizer, SFaceRecognizer):
                    name, score = recognizer.recognize(aligned)
                else:
                    gray_crop = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    name, score = recognizer.recognize(gray_crop)

                x, y, w, h = det.box
                if name:
                    attendance.mark(name)
                    color = (0, 255, 0)
                    label = name
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


def run_capture():
    name = input("Enter the person's name: ").strip().capitalize()
    if not name:
        print("[main] Name cannot be empty.")
        return
    detector = create_detector()
    capture_faces(name, detector)


def main():
    parser = argparse.ArgumentParser(description="AI Attendance System")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--capture", action="store_true", help="capture a new person's face images")
    group.add_argument("-t", "--train", action="store_true", help="(re)train the recognition model")
    args = parser.parse_args()

    if args.capture:
        run_capture()
    elif args.train:
        train_model()
    else:
        run_recognition()


if __name__ == "__main__":
    main()