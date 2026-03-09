import cv2
from ultralytics import YOLO
import time
import serial  # 安装: pip install pyserial

# 初始化串口（根据实际端口修改）
arduino = serial.Serial('/dev/ttyUSB0', 9600)  # Linux
# arduino = serial.Serial('COM3', 9600)        # Windows

# 加载YOLOv8模型
model = YOLO("yolov8n.pt")

# 定义目标规则
TARGET_RULES = {
    "person": {
        "min_confidence": 0.7,
        "cooldown": 5,
        "last_trigger": 0,
        "action": lambda frame, box: (
            arduino.write(b'1'),  # 发送'1'给Arduino
            print("已发送信号到Arduino")
        )
    }
}

# 摄像头捕获
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8检测
        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                cls_name = class_names[cls_id]

                if cls_name in TARGET_RULES:
                    rule = TARGET_RULES[cls_name]
                    if conf >= rule["min_confidence"]:
                        current_time = time.time()
                        if current_time - rule["last_trigger"] > rule["cooldown"]:
                            rule["last_trigger"] = current_time
                            rule["action"](frame, box)  # 触发Arduino

        # 显示画面（可选）
        cv2.imshow("YOLOv8 + Arduino控制", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()