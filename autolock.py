import cv2
import yolov5
import numpy as np
import mediapipe as mp
import ctypes
import time
import os

# Define the path to the YOLOv5 model on the user's desktop
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(user_desktop, "yolov5l.pt")

# Load the YOLOv5 model
model = yolov5.load(model_path)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize webcam
cap = cv2.VideoCapture(0)

# Configuration for grace period
consecutive_misses = 0
grace_period_threshold = 3  # Number of consecutive misses allowed before locking
locked = False  # Track whether the computer is currently locked

def detect_person_and_face():
    global consecutive_misses, locked

    while True:
        # Capture frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image from webcam")
            break

        # Convert the image from BGR to RGB for YOLOv5 and Mediapipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform YOLOv5 inference to detect objects
        results = model(img_rgb)

        # Assume no person is detected
        person_detected = False

        # Process each detected object
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label_text = model.names[int(cls)]
            
            # If a person is detected
            if label_text == "person":
                print("Person found")
                person_detected = True

                # Extract the person region for face detection
                person_region = img_rgb[y1:y2, x1:x2]

                # Detect face using Mediapipe within the person region
                with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                    face_results = face_detection.process(person_region)

                    if face_results.detections:
                        print("Face detected, access granted")
                        consecutive_misses = 0  # Reset the miss counter
                        locked = False  # Reset lock status
                    else:
                        print("Face not detected")
                        consecutive_misses += 1

        # If no person is detected, increase the miss counter
        if not person_detected:
            print("Person not detected")
            consecutive_misses += 1

        # Lock the computer if the grace period threshold is exceeded
        if consecutive_misses >= grace_period_threshold and not locked:
            print("Locking computer due to repeated misses")
            ctypes.windll.user32.LockWorkStation()  # Lock the computer
            locked = True  # Mark the computer as locked
            consecutive_misses = 0  # Reset after locking

        # Wait for 1 second before the next check
        time.sleep(1)

# Main entry point
if __name__ == "__main__":
    try:
        detect_person_and_face()
    finally:
        # Ensure the webcam is released and windows are closed on exit
        cap.release()
        cv2.destroyAllWindows()
