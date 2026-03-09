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
            self.model(dummy_frame, imgsz=320, verbose=False)
        print("模型预热完成")

    def reset_detector(self):
        """重置检测器状态"""
        self.last_print_time = 0

    def detect_target(self, confidence_thresh=0.5, duration=5):
        """纯检测函数，不执行任何动作"""
        self.reset_detector()
        self.camera_manager.clear_buffer()  # 清除摄像头缓冲

        cap = self.camera_manager.get_camera()
        start_time = time.time()
        detected_target = None
        highest_conf = 0

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取视频帧")
                    break

                # 使用imgsz=320进行推理
                results = self.model(frame, imgsz=320, verbose=False)
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

                            if conf >= confidence_thresh and cls_id < len(self.class_names):
                                print(f"[检测] 发现目标: {self.class_names[cls_id]} "
                                      f"(ID: {cls_id}, 置信度: {conf:.2f})")
                                # 只记录置信度最高的目标
                                if conf > highest_conf:
                                    highest_conf = conf
                                    detected_target = cls_id

                time.sleep(0.05)
        finally:
            pass
        return detected_target


class RobotTracking:
    def __init__(self):
        self.camera_manager = CameraManager()
        self.camera_manager.initialize()  # 确保摄像头已初始化
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

                line_pid = PID(Kp=-2.35, Kd=0, outmax=400, outmin=-400)
                pid_out = line_pid.Calc(diff, 0)
                print(f"偏差: {diff}, PID输出: {pid_out}")
                self.mover.any_ward(speed=80, turn=pid_out, times=50)
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
            model_path="aabest_ncnn_model",
            class_names=["person", "criminal", "tank", "car", "cloth", "stone", "code"]
        )
        # 预初始化跟踪器
        self.tracker = RobotTracking()
        print("所有组件初始化完成")

        # 添加等待按键功能
        self._wait_for_key_press()

    def _wait_for_key_press(self):
        """等待用户按键开始执行"""
        print("\n所有准备工作已完成，准备开始执行...")
        print("请按下键盘上的 's' 键开始执行程序，或按 'q' 键退出")

        # 使用OpenCV创建一个简单的窗口来捕获按键
        cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Control", 300, 100)

        while True:
            # 显示提示信息
            img = np.zeros((100, 300, 3), np.uint8)
            cv2.putText(img, "Press 's' to start", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(img, "or 'q' to quit", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Control", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 按下's'键开始
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):  # 按下'q'键退出
                cv2.destroyAllWindows()
                print("程序已退出")
                exit(0)

        print("程序开始执行...")

    def execute_action(self, target_id):
        """独立的动作执行函数"""
        actions = {
            0: lambda: self.mv.stop(),
            1: lambda: self.mv.stop(),  # criminal
            2: lambda: self.mv.stop(),  # tank
            3: lambda: self.mv.stop(),  # car
            4: lambda: self.mv.reset(),  # cloth
            5: lambda: self.mv.prepare(),  # stone
            6: lambda: self.mv.wave_hands()  # code
        }
        if target_id in actions:
            actions[target_id]()
            time.sleep(1)  # 确保动作完成

    def part1(self):
        """二维码"""
        print("执行第一部分: 二维码识别")
        self.mv.move_forward(speed=20, times=2300)
        time.sleep(2.3)
        self.mv.move_left(speed=20, times=2300)
        time.sleep(2.4)
        self.mv.turn_left(speed=20, times=1000)
        time.sleep(1)

        print("等待检测二维码...")
        detected = self.detector.detect_target(duration=5)
        if detected == 6:  # code是6
            self.execute_action(6)
        else:
            print("未检测到二维码，执行默认动作")
            self.mv.wave_hands()
            time.sleep(1.3)

    def part2(self):
        """拳头或者布"""
        print("执行第二部分: 拳头或布识别")
        self.mv.move_backward(speed=20, times=4600)
        time.sleep(4.8)
        self.mv.turn_right(speed=20, times=2000)
        time.sleep(2)

        print("等待检测拳头或布...")
        detected = self.detector.detect_target(duration=5)
        if detected == 5:  # stone是5
            self.execute_action(5)
        elif detected == 4:  # cloth是4
            self.execute_action(4)
        else:
            print("未检测到拳头或布，执行默认动作")
            self.execute_action(5)  # 默认执行石头动作

    def part3(self):
        """击打坦克"""
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
        detected = self.detector.detect_target(duration=5)
        if detected == 2:  # tank是2
            self.execute_action(2)
        else:
            print("未检测到坦克，执行默认动作")
            self.execute_action(2)

    def part4(self):
        """击打罪犯"""
        print("执行第四部分: 击打罪犯")
        self.mv.move_right(speed=20, times=2300)
        time.sleep(2.4)
        self.mv.turn_right(speed=20, times=2000)
        time.sleep(2)

        print("等待检测罪犯...")
        detected = self.detector.detect_target(duration=5)
        if detected == 1:  # criminal是1
            self.execute_action(1)
        else:
            print("未检测到罪犯，执行默认动作")
            self.execute_action(1)

    def part5(self):
        """巡线"""
        print("执行第五部分: 巡线")
        self.mv.move_forward(speed=20, times=2200)
        time.sleep(2.1)
        self.tracker.line_tracking()
        time.sleep(3)

    def run_sequence(self):
        print("开始执行完整流程...")
        try:
            self.part1()
            self.part2()
            self.part3()
            self.part4()
            self.part5()
            print("流程执行完毕")
        except Exception as e:
            print(f"程序运行出错: {str(e)}")
        finally:
            CameraManager().release_camera()


if __name__ == '__main__':
    controller = RobotController()
    controller.run_sequence()

    # print("开始单独巡线")
    # tracker = RobotTracking()
    # try:
    #     tracker.line_tracking()
    # except KeyboardInterrupt:
    #     print("巡线已停止")
    # finally:
    #     CameraManager().release_camera()