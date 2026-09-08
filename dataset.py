import os
import cv2

import config
from face_detector import align_face


def capture_faces(name, detector, count=config.CAPTURE_IMAGE_COUNT, camera_index=None):
    camera_index = config.CAMERA_INDEX if camera_index is None else camera_index
    person_dir = os.path.join(config.IMAGES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {camera_index}).")

    saved = 0
    print(f"[dataset] Capturing up to {count} images for '{name}'. Press 'q' to stop early.")

    try:
        while saved < count:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            display = frame.copy()

            for det in detections:
                aligned = align_face(frame, det)
                if aligned is None:
                    continue
                saved += 1
                out_path = os.path.join(person_dir, f"{name}_{saved}.jpg")
                cv2.imwrite(out_path, aligned)

                x, y, w, h = det.box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, f"Image {saved}/{count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if saved >= count:
                    break

            cv2.imshow("Capture Faces (q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[dataset] Captured {saved} images for '{name}'.")
    return saved