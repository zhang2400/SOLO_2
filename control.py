import cv2
from ultralytics import YOLO
import time
import json
from concurrent.futures import ThreadPoolExecutor
from robotpi_movement import Movement


class DetectionController:
    def __init__(self):
        self.model = YOLO('best_ncnn_model', task='detect')
        self.mover = Movement()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_action_time = 0
        self.ACTION_COOLDOWN = 2.0

        self.CLASSES = ["person", "criminal", "tank", "car", "cloth", "bone", "code"]
        self.ACTIONS = {
            0: lambda: self.mover.stop(),
            1: lambda: self.mover.stop(),
            2: lambda: self.mover.stop(),
            3: lambda: self.mover.stop(),
            4: lambda: self.mover.reset(),
            5: lambda: self.mover.prepare(),
            6: lambda: self.mover.wave_hands()
        }

    def async_action(self, action_func):
        self.executor.submit(action_func)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        current_time = time.time()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)

                if conf > 0.7 and (current_time - self.last_action_time) >= self.ACTION_COOLDOWN:
                    detections.append({
                        "class": self.CLASSES[class_id],
                        "confidence": conf,
                        "timestamp": current_time
                    })
                    self.async_action(self.ACTIONS[class_id])
                    self.last_action_time = current_time

        return detections

    def run(self):
        cap = cv2.VideoCapture("/dev/video0")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.process_frame(frame)
                if results:
                    print(json.dumps(results, indent=2))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown()


if __name__ == "__main__":
    controller = DetectionController()
    controller.run()
