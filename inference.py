import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. I'm loading the 'brain' we just trained
model = load_model('action.keras')
actions = np.array(['HELLO', 'THANKS'])

# 2. Setting up the Vision Engine
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 3. I'm creating a buffer to hold the last 30 frames of data
sequence = []
sentence = []
threshold = 0.8 # Only show result if confidence is > 80%

cap = cv2.VideoCapture(0)

print("Final Test Run Started! Show me your gestures.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Extracting landmarks
    keypoints = np.zeros(21 * 3) 
    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        
        # Drawing for feedback
        for lm in landmarks:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # I'm adding the current frame's data to my sequence
    sequence.append(keypoints)
    sequence = sequence[-30:] # Keep only the last 30 frames

    # 4. If I have enough data (30 frames), I'll make a prediction
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        
        # If the AI is confident enough, I'll update the screen
        if res[np.argmax(res)] > threshold:
            action_name = actions[np.argmax(res)]
            
            # Simple logic to only show the word if it's different from the last one
            if len(sentence) > 0:
                if action_name != sentence[-1]:
                    sentence.append(action_name)
            else:
                sentence.append(action_name)

        if len(sentence) > 5: # Keep the last 5 words
            sentence = sentence[-5:]

    # Visualizing the recognized words
    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Gesture Recognition Test Run', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
