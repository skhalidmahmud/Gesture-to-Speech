import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# I'm using the modern Hand Landmarker API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Modern Vision Engine Started (Python 3.12 Stable).")
print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        cv2.putText(frame, "HAND DETECTED!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for landmarks in detection_result.hand_landmarks:
            # I'm drawing the points manually to ensure it works on every system
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
            # I'll also draw lines between the joints to make it look like a skeleton
            connections = [
                (0,1), (1,2), (2,3), (3,4), # Thumb
                (0,5), (5,6), (6,7), (7,8), # Index
                (5,9), (9,10), (10,11), (11,12), # Middle
                (9,13), (13,14), (14,15), (15,16), # Ring
                (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
            ]
            for start_idx, end_idx in connections:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                pt1 = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                pt2 = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Looking for hand...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Modern AI Vision Test', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

detector.close()
cap.release()
cv2.destroyAllWindows()
