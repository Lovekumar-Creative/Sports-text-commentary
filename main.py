import cv2
import torch
from ultralytics import YOLO

from tracker import get_objects
from possession import get_ball_owner
from commentary import generate_commentary


device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained model → player detection
player_model = YOLO("yolov8n.pt")

# Fine tuned model → football detection
ball_model = YOLO(r"D:\Github\Sports text commentary\runs\detect\train2\weights\best.pt")

cap = cv2.VideoCapture("input.mp4")

previous_owner = None
commentary_file = open("commentary.txt", "w")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,360))

    # Detect players
    player_results = player_model.track(frame, persist=True)

    # Detect football
    ball_results = ball_model.predict(frame)

    players = []
    ball = None

    # -------------------------
    # Player Detection
    # -------------------------
    for result in player_results:

        boxes = result.boxes

        if boxes.id is None:
            continue

        for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):

            if int(cls) != 0:   # COCO class 0 = person
                continue

            x1,y1,x2,y2 = map(int,box)
            track_id = int(track_id)

            players.append((track_id,(x1,y1,x2,y2)))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,
                        f"Player {track_id}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

    # -------------------------
    # Football Detection
    # -------------------------
    for result in ball_results:

        boxes = result.boxes

        for box in boxes.xyxy:

            x1,y1,x2,y2 = map(int,box)

            ball = (x1,y1,x2,y2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,
                        "Football",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2)

    # -------------------------
    # Ball Possession
    # -------------------------
    owner = get_ball_owner(players, ball)

    # -------------------------
    # Commentary
    # -------------------------
    comment = generate_commentary(previous_owner, owner)

    if comment:
        print(comment)
        commentary_file.write(comment + "\n")

    previous_owner = owner

    cv2.imshow("Football Commentary AI",frame)

    if cv2.waitKey(1) == ord("q"):
        break

commentary_file.close()
cap.release()
cv2.destroyAllWindows()