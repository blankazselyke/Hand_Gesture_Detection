import cv2
import mediapipe as mp
import math
import numpy as np

cap_width = 1000
cap_height = 1000
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

saved = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    gesture = []

    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for landmark in hand_landmarks.landmark:
                x_coordinate = int(landmark.x * cap_width)
                y_coordinate = int(landmark.y * cap_height)
                gesture.append((x_coordinate, y_coordinate))
            print(gesture)

    cv2.imshow('Hand Gesture', frame)
    key = cv2.waitKey(100)

    if key & 0xFF == ord('q') or key == 27:
        break


cap.release()
cv2.destroyAllWindows()
