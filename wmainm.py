import time
import cv2
import numpy as np
from ultralytics import YOLO
from robotpi_movement import Movement
from pid import PID
from circle_detect import LineTracker


class CameraManager:
    """集中管理摄像头资源的单例类"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cap = None
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, index=0, width=480, height=480):
        """预初始化摄像头"""
        if not self._initialized:
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            self.cap.set(3, width)
            self.cap.set(4, height)
            # 预热摄像头
            for _ in range(10):
                self.cap.read()
            self._initialized = True

    def get_camera(self):
        if not self._initialized:
            self.initialize()
        return self.cap

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self._initialized = False

    def clear_buffer(self):
        """清除摄像头缓冲区"""
        if self.cap is not None and self.cap.isOpened():
            for _ in range(5):  # 清除5帧缓冲
                self.cap.grab()


class TerminalYOLODetector:
    def __init__(self, model_path, class_names):
        # 预加载模型
        print("正在预加载YOLO模型...")
        self.model = YOLO(model_path, task='detect')
        # 预热模型
        self._warm_up_model()
        self.class_names = class_names
        self.last_print_time = 0
        self.print_interval = 1.0
        self.mover = Movement()
        self.camera_manager = CameraManager()
        # 预初始化摄像头
        self.camera_manager.initialize()

    def _warm_up_model(self):
        """预热模型，让模型完成初始加载"""
        print("正在预热YOLO模型...")
        dummy_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        for _ in range(3):  # 运行几次推理预热
            self.model(dummy_frame, verbose=False)
        print("模型预热完成")

    def reset_detector(self):
        """重置检测器状态"""
        self.last_print_time = 0

    def detect_target_with_position(self, target_class, confidence_thresh=0.5, duration=5):
        """检测目标并返回位置信息"""
        self.reset_detector()
        self.camera_manager.clear_buffer()

        cap = self.camera_manager.get_camera()
        start_time = time.time()
        detected_target = None
        highest_conf = 0
        target_position = None  # 'left', 'right' or None
        frame_center = 240  # 假设画面宽度为480

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取视频帧")
                    break

                results = self.model(frame, verbose=False)
                current_time = time.time()

                if current_time - self.last_print_time >= self.print_interval:
                    self.last_print_time = current_time

                    for result in results:
                        boxes = result.boxes
                        if len(boxes) == 0:
                            print("[状态] 未检测到目标")
                            continue

                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            box_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2  # 计算目标中心x坐标

                            if (conf >= confidence_thresh and cls_id < len(self.class_names) and
                                    self.class_names[cls_id] == target_class):
                                print(f"[检测] 发现目标: {self.class_names[cls_id]} "
                                      f"(ID: {cls_id}, 置信度: {conf:.2f}, 位置: {'左' if box_center < frame_center else '右'})")

                                if conf > highest_conf:
                                    highest_conf = conf
                                    detected_target = cls_id
                                    target_position = 'left' if box_center < frame_center else 'right'

                time.sleep(0.05)
        finally:
            pass
        return detected_target, target_position


class RobotTracking:
    def __init__(self):
        self.camera_manager = CameraManager()
        self.camera_manager.initialize()
        self.cap = self.camera_manager.get_camera()
        self.line = LineTracker()
        self.mover = Movement()

    def line_tracking(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("视频帧获取错误")
                    break

                self.line.line_process(frame)
                diff = self.line.deviation

                line_pid = PID(Kp=-2.4, Kd=0, outmax=400, outmin=-400)
                pid_out = line_pid.Calc(diff, 0)
                print(f"偏差: {diff}, PID输出: {pid_out}")
                self.mover.any_ward(speed=60, turn=pid_out, times=50)
        finally:
            pass


class RobotController:
    def __init__(self):
        # 预初始化所有组件
        print("正在预初始化所有组件...")
        self.mv = Movement()
        # 预初始化摄像头
        CameraManager().initialize()
        # 预初始化检测器
        self.detector = TerminalYOLODetector(
            model_path="best_ncnn_model",
            class_names=["person", "criminal", "tank", "car", "cloth", "stone", "code"]
        )
        # 预初始化跟踪器
        self.tracker = RobotTracking()
        print("所有组件初始化完成")

    def execute_action(self, target_id, position=None):
        """独立的动作执行函数，增加位置参数"""
        actions = {
            0: lambda: self.mv.stop(),
            1: lambda: self._attack_criminal(position),  # criminal
            2: lambda: self._attack_tank(position),  # tank
            3: lambda: self.mv.stop(),  # car
            4: lambda: self.mv.reset(),  # cloth
            5: lambda: self.mv.prepare(),  # stone
            6: lambda: self.mv.wave_hands()  # code
        }
        if target_id in actions:
            actions[target_id]()
            time.sleep(1)  # 确保动作完成

    def _attack_tank(self, position):
        """根据坦克位置执行不同的攻击动作"""
        if position == 'left':
            print("坦克在左侧，执行左转攻击")
            self.mv.turn_left(speed=30, times=1500)
            self.mv.move_forward(speed=50, times=2000)
        elif position == 'right':
            print("坦克在右侧，执行右转攻击")
            self.mv.turn_right(speed=30, times=1500)
            self.mv.move_forward(speed=50, times=2000)
        else:
            print("坦克位置未知，执行默认攻击")
            self.mv.move_forward(speed=50, times=2000)

    def _attack_criminal(self, position):
        """根据罪犯位置执行不同的攻击动作"""
        if position == 'left':
            print("罪犯在左侧，执行左转攻击")
            self.mv.turn_left(speed=30, times=1000)
            self.mv.move_forward(speed=50, times=1500)
        elif position == 'right':
            print("罪犯在右侧，执行右转攻击")
            self.mv.turn_right(speed=30, times=1000)
            self.mv.move_forward(speed=50, times=1500)
        else:
            print("罪犯位置未知，执行默认攻击")
            self.mv.move_forward(speed=50, times=1500)

    # ... 其他part方法保持不变 ...

    def part3(self):
        """击打坦克（改进版）"""
        print("执行第三部分: 击打坦克")
        self.mv.turn_left(speed=20, times=1000)
        time.sleep(1)
        self.mv.move_left(speed=20, times=2300)
        time.sleep(2.4)
        self.mv.move_forward(speed=20, times=2300)
        time.sleep(2.4)
        self.mv.turn_left(speed=20, times=1000)
        time.sleep(1)

        print("等待检测坦克...")
        detected, position = self.detector.detect_target_with_position("tank", duration=5)
        if detected == 2:  # tank是2
            self.execute_action(2, position)
        else:
            print("未检测到坦克，执行默认动作")
            self.execute_action(2)

    def part4(self):
        """击打罪犯（改进版）"""
        print("执行第四部分: 击打罪犯")
        self.mv.move_right(speed=20, times=2300)
        time.sleep(2.4)
        self.mv.turn_right(speed=20, times=2000)
        time.sleep(2)

        print("等待检测罪犯...")
        detected, position = self.detector.detect_target_with_position("criminal", duration=5)
        if detected == 1:  # criminal是1
            self.execute_action(1, position)
        else:
            print("未检测到罪犯，执行默认动作")
            self.execute_action(1)

    # ... 其他方法保持不变 ...


if __name__ == '__main__':
    controller = RobotController()
    controller.run_sequence()