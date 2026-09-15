import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DATA_PATH = "data"
actions = ["Hello", "Yes", "No", "Thanks", "ILoveYou", "Good"]
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

            # Only take the first detected hand's 63 keypoints, matching
            # main.py's inference-time extraction exactly. Concatenating
            # every detected hand (as this used to do) produces 126+ values
            # whenever a second hand strays into frame, which silently fails
            # train.py's fixed-length check and drops the whole sequence.
            keypoints = np.zeros(63)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                temp = []
                for lm in hand_landmarks.landmark:
                    temp.extend([lm.x, lm.y, lm.z])
                if len(temp) == 63:
                    keypoints = temp

            npy_path = os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy")
            np.save(npy_path, keypoints)

            cv2.putText(image, f'{action} {seq}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow('Collecting Data', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()