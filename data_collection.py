import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# I'm setting up the path where I'll store the data
DATA_PATH = os.path.join('MP_Data') 

# I'll record 30 videos for each gesture
no_sequences = 30
# Each video will be 30 frames long
sequence_length = 30

# Pick a gesture name here!
action = input("Enter the name of the gesture you want to record (e.g. HELLO): ").upper()

# I'm creating the folders if they don't exist
for sequence in range(no_sequences):
    try: 
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
    except:
        pass

# Setting up the MediaPipe detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print(f"I'm ready to record {action}. Get your hand ready!")

for sequence in range(no_sequences):
    for frame_num in range(sequence_length):

        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        # I'm extracting the coordinates
        keypoints = np.zeros(21 * 3) # Default to zeros if no hand found
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            # Drawing for feedback
            for lm in landmarks:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # I'm adding some text to help me stay organized while recording
        if frame_num == 0: 
            cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            cv2.waitKey(2000) # Wait 2 seconds before each video
        else: 
            cv2.putText(frame, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)

        # I'm saving the data as a numpy array
        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
        np.save(npy_path, keypoints)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Finished collecting data for {action}!")
