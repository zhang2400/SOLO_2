import numpy as np
import cv2
from robotPi import robotPi
from rev_cam import rev_cam

# 图像参数
width = 480
height = 180
channel = 1
resized_height = int(width * 0.75)  # 360

# 加载模型
model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')

# 颜色阈值 (HSV格式)
START_COLOR_LOWER = np.array([30, 50, 50])  # 起点颜色范围 (示例：绿色)
START_COLOR_UPPER = np.array([90, 255, 255])
END_COLOR_LOWER = np.array([0, 50, 50])  # 终点颜色范围 (示例：红色)
END_COLOR_UPPER = np.array([10, 255, 255])


def detect_special_markers(frame):
    """检测起点和终点标记"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 检测起点
    start_mask = cv2.inRange(hsv, START_COLOR_LOWER, START_COLOR_UPPER)
    start_contours, _ = cv2.findContours(start_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 检测终点
    end_mask = cv2.inRange(hsv, END_COLOR_LOWER, END_COLOR_UPPER)
    end_contours, _ = cv2.findContours(end_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(start_contours) > 0, len(end_contours) > 0


def line_following():
    cap = cv2.VideoCapture(0)
    robot = robotPi()
    at_start = False
    at_end = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = rev_cam(frame)
        original_frame = frame.copy()  # 保留原始帧用于标记检测

        # 图像预处理
        frame = cv2.resize(frame, (width, resized_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[resized_height - height:, :]

        # 检测起点和终点
        is_start, is_end = detect_special_markers(original_frame)

        if is_start and not at_start:
            print("到达起点!")
            at_start = True
        elif is_end and not at_end:
            print("到达终点!")
            at_end = True
            robot.movement.stop()
            break

        # 巡线逻辑
        _, prediction = model.predict(roi.reshape(1, width * height))
        value = prediction.argmax(-1)

        if value == 0:  # 前进
            robot.movement.move_forward(speed=30, times=100)
        elif value == 1:  # 左转
            robot.movement.left_ward(speed=20, angle=15, times=100)
        elif value == 2:  # 右转
            robot.movement.right_ward(speed=20, angle=15, times=100)
        elif value == 3:  # 停止
            robot.movement.stop()

        # 显示调试信息
        debug_frame = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        if is_start:
            cv2.putText(debug_frame, "START", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_end:
            cv2.putText(debug_frame, "END", (width - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Line Following", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    line_following()