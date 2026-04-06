import os

folders = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

import cv2
import os

video_path = "input.mp4"

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:   # save every 10th frame
        filename = f"frame_{saved_count}.jpg"
        cv2.imwrite(f"dataset/images/train/{filename}", frame)
        saved_count += 1

    frame_count += 1

cap.release()

print("Frames extracted:", saved_count)