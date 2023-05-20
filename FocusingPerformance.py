# Importing required packages
import argparse
import os
import random

import cv2
import dlib
import numpy as np
import pandas as pd
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

emotion_offsets = (20, 40)
emotions = {
    0: {"emotion": "Angry - Unfocused State", "color": (193, 69, 42)},
    1: {"emotion": "Distracted - Unfocused State", "color": (164, 175, 49)},
    2: {"emotion": "Fear - Unfocused State", "color": (40, 52, 155)},
    3: {"emotion": "Good Mood - Focused State", "color": (23, 164, 28)},
    4: {"emotion": "Sad - Distracted State", "color": (164, 93, 23)},
    5: {"emotion": "Surprise - Distracted State", "color": (218, 229, 97)},
    6: {"emotion": "Neutral - Focused State", "color": (108, 72, 200)},
}


def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = "models/emotionModel.hdf5"
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

main_path = "/Users/ihelshal/Kaggle/SDAIA/CogniClass/TestData/"
samples = [fle for fle in os.listdir(main_path) if fle != ".DS_Store"]

samples = random.sample(samples, 1)
print(samples)

for video_ in samples:
    cap = cv2.VideoCapture(main_path + video_)

    # Get video properties
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get video properties
    print("Input Video Properties:")
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Codec:", int(cap.get(cv2.CAP_PROP_FOURCC)))

    desired_width = 800
    aspect_ratio = cap_width / cap_height
    desired_height = int(desired_width / aspect_ratio)

    frame_skip = 1  # Process every second frame
    frame_count = 0

    frames = []  # List to store processed frames
    # Initialize the DataFrame to store the results
    results_df = pd.DataFrame(columns=["Frame Number", "Category Detected"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            results_df = results_df.append(
                {
                    "Frame Number": cap.get(cv2.CAP_PROP_POS_FRAMES),
                    "Category Detected": "Missed",
                },
                ignore_index=True,
            )
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (desired_width, desired_height))

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(grayFrame, 0)
        for rect in rects:
            shape = predictor(grayFrame, rect)
            points = shapePoints(shape)
            (x, y, w, h) = rectPoints(rect)
            grayFace = grayFrame[y : y + h, x : x + w]
            try:
                grayFace = cv2.resize(grayFace, (emotionTargetSize))
            except:
                continue

            grayFace = grayFace.astype("float32")
            grayFace = grayFace / 255.0
            grayFace = (grayFace - 0.5) * 2.0
            grayFace = np.expand_dims(grayFace, 0)
            grayFace = np.expand_dims(grayFace, -1)
            emotion_prediction = emotionClassifier.predict(grayFace)
            emotion_probability = np.max(emotion_prediction)
            if emotion_probability > 0.36:
                emotion_label_arg = np.argmax(emotion_prediction)
                color = emotions[emotion_label_arg]["color"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.line(frame, (x, y + h), (x + 20, y + h + 20), color, thickness=2)
                cv2.rectangle(
                    frame, (x + 20, y + h + 20), (x + 110, y + h + 40), color, -1
                )
                cv2.putText(
                    frame,
                    emotions[emotion_label_arg]["emotion"],
                    (x + 40, y + h + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                # Add the frame number and category to the DataFrame
                results_df = results_df.append(
                    {
                        "Frame Number": cap.get(cv2.CAP_PROP_POS_FRAMES),
                        "Category Detected": emotions[emotion_label_arg]["emotion"],
                    },
                    ignore_index=True,
                )
            else:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                results_df = results_df.append(
                    {
                        "Frame Number": cap.get(cv2.CAP_PROP_POS_FRAMES),
                        "Category Detected": "Missed",
                    },
                    ignore_index=True,
                )

        frames.append(frame)
        cv2.imshow("Emotion Recognition from Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    output_video = cv2.VideoWriter(
        "/Users/ihelshal/Kaggle/SDAIA/CogniClass/outputs/ProcessedVideos/" + video_,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (desired_width, desired_height),
    )
    for frame in frames:
        output_video.write(frame)
    output_video.release()

    name = video_.split(".")
    results_df.to_excel(
        "/Users/ihelshal/Kaggle/SDAIA/CogniClass/outputs/Results/"
        + str(name[0])
        + ".xlsx",
        index=False,
    )
