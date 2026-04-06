# ⚽ Sports Text Commentary Generation using Computer Vision

This project generates **automatic football match text commentary** using **Computer Vision and Deep Learning**.  
The system detects **players and football from a video**, tracks their movement, analyzes ball possession, and generates **real-time commentary similar to a sports commentator**.

The project uses **YOLOv8 for object detection**, tracking algorithms, and rule-based commentary generation.

---

# 🚀 Features

- Detect **players and football** in match video
- **Fine-tuned YOLOv8 model** for accurate football detection
- **Pretrained YOLO model** for player detection
- Object **tracking of players and ball**
- **Ball possession detection**
- **Automatic sports commentary generation**
- Display **processed video with commentary**
- Web interface using **Flask**
---

# 🧠 Project Workflow

The system follows these steps:

### 1. Input Video
A football match video is provided as input.

### 2. Object Detection
- **YOLOv8 pretrained model** detects:
  - Players
- **Fine-tuned YOLOv8 model** detects:
  - Football

### 3. Object Tracking
Players and ball are tracked across frames.

### 4. Possession Detection
Distance between player and ball is calculated.  
The closest player is considered the **ball owner**.

### 5. Commentary Generation
Based on ball movement and player possession.

Example commentary:

```
Player 3 passes the ball to Player 7
Player 5 gains possession of the ball
Player 2 intercepts the pass
```

### 6. Web Visualization
Processed video and live commentary are displayed on a webpage.

---

# 🏗️ Project Structure

```
sports-commentary-ai
│
├── app.py                # Flask web application
├── detector.py           # YOLO detection logic
├── tracker.py            # Object tracking
├── possession.py         # Ball possession detection
├── commentary.py         # Commentary generation logic
│
├── templates
│   └── index.html        # Web interface
│
├── input.mp4             # Input football video
├── output.mp4            # Processed output video
│
├── runs                  # Training results (YOLO)
├── models
│   └── football.pt       # Fine-tuned football detection model
│
└── README.md
```

---

# ⚙️ Technologies Used

- Python
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Flask
- HTML / CSS / JavaScript

---

# 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics opencv-python flask torch
```

---

# ▶️ Running the Project

Run the Flask server:

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

The webpage will display:

- Processed football video
- Generated sports commentary

---

# 🎯 Model Training (Football Detection)

The football detection model was fine-tuned using a custom dataset.

Training command:

```bash
yolo detect train \
model=yolov8n.pt \
data=data.yaml \
epochs=50 \
imgsz=640
```

This improves detection accuracy for **small football objects**.

---

# 🧪 Example Commentary Output

```
Player 4 gains possession of the ball.
Player 4 passes the ball to Player 8.
Player 8 moves forward with the ball.
Player 10 intercepts the pass.
```

---

# 📊 Future Improvements

- Real **player identification**
- **Team detection**
- **Goal detection**
- **Advanced NLP-based commentary**
- **Audio commentary generation**
- **Live match streaming support**

---

# 👨‍💻 Author

**Love Kumar**  
B.Tech Computer Science  
Lovely Professional University
