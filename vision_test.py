import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# I'm setting up the hand landmarker options
# I'm using the 'hand_landmarker.task' file I just downloaded
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# Initializing the detector
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Modern Vision Engine Started (Python 3.14 Compatible).")
print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    # Flipping and preparing the image
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Converting to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # I'm running the detection
    detection_result = detector.detect(mp_image)

    # If I find hands, I'll draw the landmarks manually
    if detection_result.hand_landmarks:
        cv2.putText(frame, "HAND DETECTED!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for landmarks in detection_result.hand_landmarks:
            for landmark in landmarks:
                # Converting normalized coordinates to pixel coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    else:
        cv2.putText(frame, "Looking for hand...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Modern AI Vision Test', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

detector.close()
cap.release()
cv2.destroyAllWindows()
