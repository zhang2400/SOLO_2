import cv2
from ultralytics import YOLO
import torch
import os

# 修复Qt问题
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载模型到指定设备
model = YOLO("yolov8n.pt").to(device)

# 打开摄像头（无显示）
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用GPU加速推理
    results = model(frame, device=device)  # 显式指定设备

    # 打印检测信息（不显示画面）
    for result in results:
        print(f"检测到 {len(result.boxes)} 个对象")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()