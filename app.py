from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import torch
from ultralytics import YOLO

from possession import get_ball_owner
from commentary import generate_commentary

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

player_model = YOLO("yolov8n.pt")
ball_model = YOLO("runs/detect/train2/weights/best.pt")

video_path = "static/input1.mp4"

running = False
paused = False
previous_owner = None

commentary_list = []


def generate_frames():

    global running, paused, previous_owner

    cap = cv2.VideoCapture(video_path)

    while True:

        if not running:
            continue

        if paused:
            continue

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640,360))

        player_results = player_model.track(frame, persist=True)
        ball_results = ball_model.predict(frame)

        players = []
        ball = None

        for result in player_results:

            boxes = result.boxes

            if boxes.id is None:
                continue

            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):

                if int(cls) != 0:
                    continue

                x1,y1,x2,y2 = map(int,box)
                track_id = int(track_id)

                players.append((track_id,(x1,y1,x2,y2)))

                #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"Player {track_id}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        for result in ball_results:

            for box in result.boxes.xyxy:

                x1,y1,x2,y2 = map(int,box)
                ball = (x1,y1,x2,y2)

                #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                #cv2.putText(frame,"Ball",(x1,y1-10),
                #            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        owner = get_ball_owner(players, ball)

        comment = generate_commentary(previous_owner, owner)

        if comment:
            commentary_list.append(comment)

        previous_owner = owner

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start")
def start():
    global running, paused
    running = True
    paused = False
    return jsonify({"status":"started"})


@app.route("/pause")
def pause():
    global paused
    paused = True
    return jsonify({"status":"paused"})


@app.route("/resume")
def resume():
    global paused
    paused = False
    return jsonify({"status":"resumed"})


@app.route("/commentary")
def commentary():
    return jsonify(commentary_list)


@app.route("/download")
def download():

    with open("commentary.txt","w") as f:
        for line in commentary_list:
            f.write(line+"\n")

    return send_file("commentary.txt",as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)