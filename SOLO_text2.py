
# from ultralytics import YOLO

# a1 = YOLO('best.pt')

# a1('v1.mp4', show=True,save=True)

from ultralytics import YOLO  # type: ignore
import cv2                    # type: ignore
import apriltag

# 加载预训练的YOLOv8模型（可以选择不同大小的模型，如yolov8n.pt, yolov8s.pt等）
model = YOLO('runs/detect/train6/weights/best_ncnn_model')  # 加载官方模型或自定义模型

# 打开摄像头（0通常是默认摄像头）
cap = cv2.VideoCapture(1)

while cap.isOpened():
    # 读取摄像头帧
    success, frame = cap.read()

    if success:
        # 在帧上运行YOLOv8推理
        results = model(frame)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带标注的帧
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()