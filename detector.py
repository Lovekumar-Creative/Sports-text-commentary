from ultralytics import YOLO

class Detector:

    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect(self, frame, device):

        results = self.model.track(
            frame,
            persist=True,
            classes=[0,32],
            conf=0.25,
            tracker="bytetrack.yaml",
            device=device
        )

        return results