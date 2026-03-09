import cv2
from ultralytics import YOLO
import os


def init_detector(model_path='best.pt'):
    """初始化检测器"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    return YOLO(model_path).cpu()


def detect_objects(cap, model, frame_skip=3, conf_threshold=0.5, max_detections=3):
    """
    执行目标检测并返回检测到的目标名称

    参数:
        cap: cv2.VideoCapture对象
        model: YOLO模型
        frame_skip: 跳帧数(默认3帧处理1次)
        conf_threshold: 置信度阈值(默认0.5)
        max_detections: 最大检测数量(默认3)

    返回:
        list: 检测到的目标名称列表
        numpy.ndarray: 原始帧图像
    """
    # 跳帧处理
    for _ in range(frame_skip):
        cap.grab()
    ret, frame = cap.retrieve()

    if not ret:
        return [], None

    # 执行检测
    results = model(frame,
                    imgsz=160,
                    conf=conf_threshold,
                    max_det=max_detections,
                    verbose=False)

    # 提取检测到的目标名称
    detected_names = []
    if len(results[0]) > 0:
        detected_names = [model.names[int(box.cls.item())]
                          for box in results[0].boxes
                          if box.conf.item() >= conf_threshold]

    return detected_names, frame


# 使用示例
if __name__ == "__main__":
    # 初始化
    model = init_detector('best.pt')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    try:
        while True:
            # 调用检测函数
            names, frame = detect_objects(cap, model)

            # 打印检测结果
            if names:
                print(f"检测到目标: {', '.join(names)}")

            # 显示画面(可选)
            if frame is not None:
                cv2.imshow('Detection', frame)
                if cv2.waitKey(10) == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()