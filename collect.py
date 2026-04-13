import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DATA_PATH = "data"
actions = ["Hello", "Thanks", "Yes", "No"]
no_sequences = 20
sequence_length = 30

os.makedirs(DATA_PATH, exist_ok=True)

for action in actions:
    for seq in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

cap = cv2.VideoCapture(0)

for action in actions:
    for seq in range(no_sequences):
        print(f"Collecting {action} sequence {seq}")
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            keypoints = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints = np.zeros(63)

            npy_path = os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy")
            np.save(npy_path, keypoints)

            cv2.putText(image, f'{action} {seq}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow('Collecting Data', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()