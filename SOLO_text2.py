from ultralytics import YOLO

a1 = YOLO('yolov8n.pt')

a1('v3.mp4', show=True,save=True)