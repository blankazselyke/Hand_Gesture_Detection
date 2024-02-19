import cv2
import mediapipe as mp
import math
import numpy as np
import csv
import tensorflow as tf


def save(code, filename, normalized):
    with open(filename, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        row = []
        row.append(code)
        for coordinates in normalized:
            row.append(coordinates[0])
            row.append(coordinates[1])
        csvwriter.writerow(row)


cap_width = 1000
cap_height = 1000
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
loaded_model = tf.keras.models.load_model('rps.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
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
            # print("Before: ", gesture)
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
            # print("After: ", vectorized)
            # print("Maximum: ", max_coord)
            # print("Try: ", vectorized)
            normalized = []
            for coordinates in vectorized:
                norm_x = round(coordinates[0] / max_coord, 4)
                norm_y = round(coordinates[1] / max_coord, 4)
                normalized.append((norm_x, norm_y))
            # print("Normalized: ", normalized)

            # Assuming test_data[0] is your single data point
            # Add an extra dimension to match the expected input shape of the model
            norm = normalized[0:22]
            test_data_single = np.expand_dims(norm, axis=0)

            # Now make predictions
            reshaped_test_data = test_data_single.reshape(1, -1)
            predictions = loaded_model.predict(reshaped_test_data)
            print(predictions)
            # Extracting predicted class labels
            predicted_labels = np.argmax(predictions, axis=1)
            print(predicted_labels)

            text = ''
            if predicted_labels == 0:
                text = 'Rock'
            elif predicted_labels == 1:
                text = 'Paper'
            else:
                text = 'Scissors'

            # Define the font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White color in BGR format
            thickness = 2

            # Define the position to write text (bottom-left corner)
            position = (50, 50)  # Example position, adjust as needed

            # Write the text on the image
            cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

    cv2.imshow('Hand Gesture', frame)
    key = cv2.waitKey(100)

    if key & 0xFF == ord('0'):  # rock
        save('0', 'rock_paper_scissors.csv', normalized)

    if key & 0xFF == ord('1'):  # paper
        save('1', 'rock_paper_scissors.csv', normalized)

    if key & 0xFF == ord('2'):  # scissors
        save('2', 'rock_paper_scissors.csv', normalized)

    if key & 0xFF == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
