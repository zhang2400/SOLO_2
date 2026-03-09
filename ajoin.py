import cv2
import time
import math
import sys
import numpy as np
from robotpi_movement import Movement
from pid import PID
from ultralytics import YOLO
from b import Robot
from line_tracker import LineTracker

global c = 0
# 根据操作系统选择不同的按键检测方式
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty
    import select

class VisionRobot:
    def __init__(self):
        # 硬件初始化
        self.mv = Movement()
        self.robot = Robot()
        self.line = LineTracker()


        # 预加载模型
        print("正在加载YOLO模型...")
        self.model = YOLO("best_ncnn_model", task='detect')
        # 预热模型
        dummy_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy_frame, imgsz=256, verbose=False)

        # PID控制器
        self.cross_pid = PID(Kp=5.5, Ki=0.01, Kd=1.5, outmax=80, outmin=-80)
        self.tank_pid = PID(Kp=-40, Ki=0.01, Kd=0.4, outmax=80, outmin=15)
        self.zuifan_pid = PID(Kp=-40, Ki=0.01, Kd=0.4, outmax=80, outmin=15)

        # 视觉标签
        self.CROSS_LABEL = "shizi"
        self.TAG_LABEL = "tag"
        self.hand_LABEL = "hand"
        self.quan_LABEL = "quantou"
        self.tank_LABEL = "tank"
        self.zuifan_LABEL = "zuifan"

        # 参数配置
        self.CONF_THRESH = 0.3
        self.CENTER_X = 240
        self.CENTER_Y = 240
        self.ALIGN_THRESHOLD = 40

        # 距离计算参数
        self.KNOWN_WIDTH = 10.0  # 十字标记的已知物理宽度(厘米)
        self.FOCAL_LENGTH = 640  # 相机焦距(像素)
        self.TARGET_DISTANCE_CM = 30  # 目标距离(厘米)
        self.DISTANCE_TO_MOVE_TIME = 14.5 # 距离与前进时间的比例系数17.5

        # 摄像头设置
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 状态机初始化状态
        self.state = "FIND_TAG"
        self._wait_for_key_press()

    def _wait_for_key_press(self):
        """等待用户按键开始执行"""
        print("\n所有准备工作已完成，准备开始执行...")
        print("请按下键盘上的 's' 键开始执行程序，或按 'q' 键退出")

        if sys.platform == 'win32':
            # Windows系统
            print("Windows系统 - 等待按键输入...")
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    print(f"检测到按键: {key}")
                    if key == 's':
                        print("程序开始执行...")
                        break
                    elif key == 'q':
                        print("程序已退出")
                        exit(0)
                time.sleep(0.1)
        else:
            # Linux/Mac系统
            print("Linux/Mac系统 - 等待按键输入...")
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while True:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1).lower()
                        print(f"检测到按键: {key}")
                        if key == 's':
                            print("程序开始执行...")
                            break
                        elif key == 'q':
                            print("程序已退出")
                            exit(0)
                    time.sleep(0.1)
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, old_settings)
                print("终端设置已恢复")

    def calculate_distance(self, box):
        """计算目标到相机的距离(单次计算)"""
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        pixel_width = x2 - x1

        if pixel_width == 0:
            return float('inf')  # 避免除以零

        # 计算距离: 距离 = (已知宽度 × 焦距) / 检测到的像素宽度
        distance = (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / pixel_width
        return distance

    def align_and_move_to_cross(self, offset=0, move_after_align=True):
        """
        十字对齐函数，对齐后计算一次距离并前进
        :param offset: 期望的中心偏移量(正负值调整)
        :param move_after_align: 是否在对齐后前进
        """
        pid = self.cross_pid

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            results = self.detect_objects(frame)
            boxes = results[0].boxes
            if not boxes:
                print("没有检测到任何物体")
                continue

            crosses = [
                (box, float(box.conf)) for box, cls in zip(boxes, boxes.cls)
                if self.model.names[int(cls)] == self.CROSS_LABEL and float(box.conf) > self.CONF_THRESH
            ]

            if not crosses:
                print("没有检测到十字")
                continue

            # 找置信度最高的十字
            best_box, best_conf = max(crosses, key=lambda x: x[1])
            x, y, w, h = best_box.xyxy[0].cpu().numpy()
            center_x = x
            error = center_x - self.CENTER_X + offset
            print(f"十字偏移: {error}, 置信度: {best_conf}")

            if abs(error) <= self.ALIGN_THRESHOLD:
                print("对齐完成，停止")
                self.mv.stop()

                if move_after_align:
                    # 计算一次距离并前进
                    distance = self.calculate_distance(best_box)
                    print(f"检测到距离: {distance:.2f}cm")

                    if distance > self.TARGET_DISTANCE_CM:
                        # 计算前进时间(简单线性关系)
                        move_time = int((distance) * self.DISTANCE_TO_MOVE_TIME)
                        move_time = max(100, min(move_time, 1200))  # 限制在100-2000ms之间
                        print(f"前进 {move_time}ms")
                        self.mv.move_forward(100, move_time)
                        time.sleep(move_time / 1000 + 0.5)  # 等待移动完成
                        # time.sleep(2)
                return True
            else:
                turn = pid.Calc(error, 0)
                print(f"PID输出: {turn}")
                self.mv.any_ward(angle=0, speed=0, turn=turn, times=40)

    def get_frame(self, discard_frames=5):
        """读取摄像头最新一帧，丢弃缓存帧"""
        frame = None
        for _ in range(discard_frames):
            ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, -1)

    def detect_objects(self, frame):
        return self.model(frame, imgsz=256)

    def process_find_tag(self, results):
        for box, cls in zip(results[0].boxes, results[0].boxes.cls):
            label = self.model.names[int(cls)]
            conf = float(box.conf)
            if label == self.TAG_LABEL and conf > self.CONF_THRESH:
                self.mv.rise_left()
                time.sleep(1.5)
                self.mv.turn_left(15, 1250)
                time.sleep(1)
                self.align_and_move_to_cross()
                self.align_and_move_to_cross()

                return True
        return False

    def process_find_hand_or_quan(self, results):
        hand_detected = False
        quan_detected = False

        for box, cls in zip(results[0].boxes, results[0].boxes.cls):
            label = self.model.names[int(cls)]
            conf = float(box.conf)
            if label == self.hand_LABEL and conf > self.CONF_THRESH:
                hand_detected = True
            elif label == self.quan_LABEL and conf > self.CONF_THRESH:
                quan_detected = True

        if hand_detected:
            print("Hand detected")
            self.mv.rise_double()
            time.sleep(2)
            self.mv.turn_right(15, 1240)
            time.sleep(1.5)
            self.align_and_move_to_cross()

            self.mv.turn_right(15, 600)
            time.sleep(1)
            self.align_and_move_to_cross()

            self.mv.turn_left(15, 620)
            time.sleep(1)
            return True
        elif quan_detected:
            print("Quantou detected")
            self.mv.rise_right()
            time.sleep(2)
            self.mv.turn_right(15, 1240)
            time.sleep(1.5)
            self.align_and_move_to_cross()

            self.mv.turn_right(15, 600)
            time.sleep(1)
            self.align_and_move_to_cross()

            self.mv.turn_left(15, 620)
            time.sleep(1)
            return True

        return False

    def process_align_tank(self, results):
        start_time = time.time()
        hit_done = False
        error = 0

        while time.time() - start_time <= 15:
            frame = self.get_frame()
            if frame is None:
                continue

            results = self.detect_objects(frame)
            for box, cls in zip(results[0].boxes, results[0].boxes.cls):
                label = self.model.names[int(cls)]
                conf = float(box.conf)

                if label == self.tank_LABEL and conf > self.CONF_THRESH:
                    x, y, w, h = box.xyxy[0].cpu().numpy()
                    tank_center_x = x
                    error = self.CENTER_X - tank_center_x
                    a = error
                    move_time = self.tank_pid.Calc(error, 0)
                    print(f"[TANK] Detected. error={error:.2f}, move_time={move_time:.2f}")

                    if abs(error) <= 20:
                        self.mv.stop()
                        time.sleep(1)
                        self.mv.move_forward(15, 250)
                        time.sleep(1)
                        self.mv.hit()
                        print("[ZUIFAN] HIT executed!")
                        time.sleep(2)
                        hit_done = True
                        break
                    else:
                        if error > 0:
                            self.mv.move_left(speed=10, times=int(move_time))
                        else:
                            self.mv.move_right(speed=10, times=int(move_time))
                else:
                    if a == 0:
                        self.mv.turn_right(15, 60)
                        a = 1
                        continue
                    else
                        a = 0
                        self.mv.turn_left(15, 120)
                        continue

            if hit_done:
                break

        print("[ZUIFAN] Executing fallback sequence...")
        self.mv.move_backward(15, 180)
        time.sleep(1)
        self.mv.turn_right(15, 610)
        time.sleep(1)
        self.align_and_move_to_cross()


        return True

    def process_align_zuifan(self, results):
        start_time = time.time()
        hit_done = False
        error = 0

        while time.time() - start_time <= 15:
            frame = self.get_frame()
            if frame is None:
                continue

            results = self.detect_objects(frame)
            for box, cls in zip(results[0].boxes, results[0].boxes.cls):
                label = self.model.names[int(cls)]
                conf = float(box.conf)

                if label == self.zuifan_LABEL and conf > self.CONF_THRESH:
                    x, y, w, h = box.xyxy[0].cpu().numpy()
                    zuifan_center_x = x
                    error = self.CENTER_X - zuifan_center_x
                    b = error
                    move_time = self.zuifan_pid.Calc(error, 0)
                    print(f"[ZUIFAN] Detected. error={error:.2f}, move_time={move_time:.2f}")

                    if abs(error) <= 20:
                        self.mv.stop()
                        time.sleep(1)
                        self.mv.move_forward(15, 260)
                        time.sleep(2)
                        self.mv.hit()
                        print("[ZUIFAN] HIT executed!")
                        time.sleep(2)
                        hit_done = True
                        break
                    else:
                        if error > 0:
                            self.mv.move_left(speed=10, times=int(move_time))
                        else:
                            self.mv.move_right(speed=10, times=int(move_time))
                else:
                    if a == 0:
                        self.mv.turn_right(15, 60)
                        a = 1
                        continue
                    else
                        a = 0
                        self.mv.turn_left(15, 120)
                        continue



            if hit_done:
                break

        print("[ZUIFAN] Executing fallback sequence...")
        self.mv.move_backward(15, 260)
        time.sleep(1)
        self.mv.turn_right(15, 670)
        time.sleep(1)
        self.align_and_move_to_cross(offset=20)  # 使用带偏移的对齐


        return True

    def xunxian(self):
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time

            ret, frame = self.cap.read()
            if not ret:
                print("video error")
                break

            self.line.line_process(frame)
            diff = self.line.deviation
            line_pid = PID(Kp=-0.8, Kd=0.01, outmax=400, outmin=-400)
            pid_out = line_pid.Calc(diff, 0)
            print(diff)
            self.mv.any_ward(speed=40, turn=pid_out, times=50)

            if elapsed_time >= 3.2 and elapsed_time < 4.2:
                self.mv.stop()
                time.sleep(0.1)
                self.align_and_move_to_cross(offset=-60)  # 最后阶段使用特定偏移


                self.mv.stop()
                break

    def run(self):
        self.align_and_move_to_cross()


        self.mv.turn_left(15, 625)
        time.sleep(1)
        # self.mv.stop()
        # time.sleep(1)
        self.align_and_move_to_cross()



        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            results = self.detect_objects(frame)

            if self.state == "FIND_TAG":
                if self.process_find_tag(results):
                    self.state = "FIND_HAND"

            elif self.state == "FIND_HAND":
                if self.process_find_hand_or_quan(results):
                    self.state = "FIND_TANK"

            elif self.state == "FIND_TANK":
                if self.process_align_tank(results):
                    self.state = "FIND_ZUIFAN"

            elif self.state == "FIND_ZUIFAN":
                if self.process_align_zuifan(results):
                    self.state = "DONE"
            elif self.state == "DONE":
                print("Mission complete!")
                break


if __name__ == "__main__":
    robot = VisionRobot()
    robot.__init__()
    robot.run()
    robot.xunxian()

    # # 初始化机器人
    # robot = VisionRobot()
    # robot.__init__()
    #
    #
    # print("=== 开始十字对齐测试 ===")
    #
    # try:
    #     # 测试标准十字对齐（偏移0）
    #     print("\n[测试1] 标准十字对齐并前进")
    #     robot.align_and_move_to_cross(offset=0, move_after_align=True)
    #     # robot.align_and_move_to_cross(offset=0, move_after_align=True)
    #     # robot.align_and_move_to_cross(offset=0, move_after_align=True)
    #
    #     # # 测试带偏移的十字对齐（偏移+20）
    #     # print("\n[测试2] 带偏移(+20)的十字对齐并前进")
    #     # robot.align_and_move_to_cross(offset=20, move_after_align=True)
    #     #
    #     # # 测试不带前进的对齐（仅对齐）
    #     # print("\n[测试3] 仅对齐不前进")
    #     # robot.align_and_move_to_cross(offset=0, move_after_align=False)
    #
    # except KeyboardInterrupt:
    #     print("\n用户中断测试")
    # finally:
    #     # 清理资源
    #     robot.mv.stop()
    #     if hasattr(robot, 'cap'):
    #         robot.cap.release()
    #     cv2.destroyAllWindows()
    #     print("测试结束，资源已释放")

    # # 初始化机器人
    # robot = VisionRobot()
    # robot.__init__()
    #
    # print("=== 开始巡线功能测试 ===")
    # print("说明：")
    # print("1. 机器人将开始巡线6.7秒")
    # print("2. 巡线结束后会执行一次十字对齐")
    # print("3. 按Ctrl+C可提前终止测试")
    #
    # try:
    #     # 单独测试巡线功能
    #     robot.xunxian()
    #
    #     print("\n巡线测试完成！")
    #     print("1. 机器人已完成6.7秒巡线")
    #     print("2. 已执行最后的十字对齐")
    #
    # except KeyboardInterrupt:
    #     print("\n用户中断测试")
    # finally:
    #     # 清理资源
    #     robot.mv.stop()
    #     if hasattr(robot, 'cap'):
    #         robot.cap.release()
    #     cv2.destroyAllWindows()
    #     print("测试结束，资源已释放")