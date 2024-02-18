import cv2
import mediapipe as mp
import math
import numpy as np
import csv

cap_width = 1000
cap_height = 1000
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

normalized = []

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
            print("Before: ", gesture)
            vectorized = []
            ref_x = gesture[0][0]
            ref_y = gesture[0][1]
            for coordinates in gesture:
                new_x = coordinates[0] - ref_x
                new_y = coordinates[1] - ref_y
                vectorized.append((new_x, new_y))
            max_coord = 0
            for coordinates in vectorized:
                for coord in coordinates:
                    if abs(coord) > max_coord:
                        max_coord = abs(coord)
            print("After: ", vectorized)
            print("Maximum: ", max_coord)
            print("Try: ", vectorized)
            normalized = []
            for coordinates in vectorized:
                norm_x = round(coordinates[0] / max_coord, 4)
                norm_y = round(coordinates[1] / max_coord, 4)
                normalized.append((norm_x, norm_y))
            print("Normalized: ", normalized)

    cv2.imshow('Hand Gesture', frame)
    key = cv2.waitKey(100)

    if key & 0xFF == ord('1'): #open 340
        # writing to csv file
        with open('gesture_data', 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            row = []
            row.append(1)
            for coordinates in normalized:
                row.append(coordinates[0])
                row.append(coordinates[1])
            csvwriter.writerow(row)

    if key & 0xFF == ord('2'): #closed 447
        # writing to csv file
        with open('gesture_data', 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            row = []
            row.append(2)
            for coordinates in normalized:
                row.append(coordinates[0])
                row.append(coordinates[1])
            csvwriter.writerow(row)

    if key & 0xFF == ord('3'): #rock 584
        # writing to csv file
        with open('gesture_data', 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            row = []
            row.append(3)
            for coordinates in normalized:
                row.append(coordinates[0])
                row.append(coordinates[1])
            csvwriter.writerow(row)

    if key & 0xFF == ord('q') or key == 27:
        break


cap.release()
cv2.destroyAllWindows()
