import cv2
import numpy as np
import time
from ropotpi_movement import Movement  # 导入您提供的运动控制类


class LineFollower:
    def __init__(self):
        # 初始化运动控制
        self.mv = Movement()

        # 摄像头设置
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 10)

        # 图像处理参数
        self.adaptive_thresh_block = 31
        self.adaptive_thresh_c = 5

        # 控制参数
        self.Kp = 0.7  # 比例系数(增大使响应更灵敏)
        self.Ki = 0.001  # 积分系数(减小防止振荡)
        self.Kd = 0.15  # 微分系数(适当阻尼)
        self.last_error = 0
        self.integral = 0

        # 运行参数
        self.base_speed = 20  # 基础前进速度(0-100)
        self.max_turn = 100  # 最大转向量
        self.lost_timeout = 2  # 丢失线路超时(秒)
        self.last_line_time = time.time()

        # 状态标志
        self.running = True

        # 初始化机器人
        self.mv.prepare()
        time.sleep(1)

    def preprocess(self, frame):
        """优化的图像预处理"""
        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, self.adaptive_thresh_block,
                                       self.adaptive_thresh_c)

        # 简化形态学操作
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return processed

    def find_line_deviation(self, processed):
        """查找黑线并计算偏离量"""
        height, width = processed.shape
        roi = processed[int(height * 0.6):, :]  # 仅使用图像下部40%

        # 垂直投影法找黑线
        histogram = np.sum(roi, axis=0)
        midpoint = width // 2

        if np.max(histogram) < 50:  # 没有检测到足够黑线像素
            return None

        # 找到左右边缘
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        # 计算中心点和归一化偏离量[-1,1]
        line_center = (left_base + right_base) // 2
        deviation = (line_center - midpoint) / midpoint

        return deviation

    def pid_control(self, error):
        """PID控制器"""
        self.integral = self.integral * 0.9 + error  # 带衰减的积分
        derivative = error - self.last_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return np.clip(output, -1, 1)

    def control_robot(self, pid_output):
        """控制机器人运动"""
        # 计算转向量
        turn = int(pid_output * self.max_turn)

        # 前进速度随转向量自动调整
        speed = max(10, self.base_speed - abs(turn) // 2)

        # 使用left_ward/right_ward方法实现平滑转向
        if pid_output > 0:  # 需要左转
            self.mv.left_ward(speed=speed, turn=turn, times=100)
        elif pid_output < 0:  # 需要右转
            self.mv.right_ward(speed=speed, turn=-turn, times=100)
        else:  # 直行
            self.mv.move_forward(speed=speed, times=100)

    def search_line(self):
        """丢失线路时的搜索行为"""
        search_start = time.time()
        while self.running and time.time() - search_start < self.lost_timeout:
            # 交替左右旋转搜索
            if self.last_error >= 0:
                self.mv.turn_left(speed=15, times=200)
            else:
                self.mv.turn_right(speed=15, times=200)

            # 检查是否重新找到线路
            ret, frame = self.cap.read()
            if ret:
                processed = self.preprocess(frame)
                deviation = self.find_line_deviation(processed)
                if deviation is not None:
                    return deviation

            time.sleep(0.1)

        return None

    def run(self):
        try:
            self.last_line_time = time.time()

            while self.running:
                start_time = time.time()

                # 读取摄像头帧
                ret, frame = self.cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break

                # 图像处理
                processed = self.preprocess(frame)
                deviation = self.find_line_deviation(processed)

                # 线路丢失处理
                if deviation is None:
                    if time.time() - self.last_line_time > self.lost_timeout:
                        print("线路丢失，启动搜索...")
                        deviation = self.search_line()
                        if deviation is None:
                            print("无法找回线路，停止")
                            self.mv.stop()
                            break

                    self.mv.stop()
                    time.sleep(0.1)
                    continue

                self.last_line_time = time.time()

                # PID控制
                pid_output = self.pid_control(deviation)

                # 控制机器人
                self.control_robot(pid_output)

                # 显示调试信息
                fps = 1 / (time.time() - start_time + 1e-6)
                print(f"FPS: {fps:.1f} | Dev: {deviation:.3f} | PID: {pid_output:.3f}")

                # 检查退出键
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.running = False
        self.cap.release()
        self.mv.stop()
        cv2.destroyAllWindows()
        print("系统已关闭")


if __name__ == "__main__":
    follower = LineFollower()
    follower.run()