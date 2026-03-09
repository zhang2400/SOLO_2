import time
import cv2
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
        return cls._instance

    def get_camera(self, index=0, width=480, height=480):
        if self.cap is None or not self.cap.isOpened():
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            self.cap.set(3, width)
            self.cap.set(4, height)
            # 给摄像头一些初始化时间
            time.sleep(1)
        return self.cap

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def clear_buffer(self):
        """清除摄像头缓冲区"""
        if self.cap is not None and self.cap.isOpened():
            for _ in range(5):  # 清除5帧缓冲
                self.cap.grab()


class TerminalYOLODetector:
    def __init__(self, model_path, class_names):
        self.model = YOLO(model_path, task='detect')
        self.class_names = class_names
        self.last_print_time = 0
        self.print_interval = 1.0
        self.mover = Movement()
        self.camera_manager = CameraManager()

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

                # 清除YOLO模型可能存在的缓存
                if hasattr(self.model, '_reset'):
                    self.model._reset()

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

                # if abs(diff) < 20:
                #     self.mover.stop()
                #     break

                line_pid = PID(Kp=-2.4, Kd=0, outmax=400, outmin=-400)
                pid_out = line_pid.Calc(diff, 0)
                print(f"偏差: {diff}, PID输出: {pid_out}")
                self.mover.any_ward(speed=80, turn=pid_out, times=50)
        finally:
            pass


class RobotController:
    def __init__(self):
        self.mv = Movement()
        self.tracker = RobotTracking()
        self.detector = TerminalYOLODetector(
            model_path="best_ncnn_model",
            class_names=["person", "criminal", "tank", "car", "cloth", "stone", "code"]
        )

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