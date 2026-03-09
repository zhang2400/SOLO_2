import cv2
import numpy as np
import time


class LineFollower:
    def __init__(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # 跟踪状态变量
        self.history = []  # 存储历史位置
        self.error_flag = False  # 错误状态标志
        self.line_center = 0  # 当前线中心
        self.deviation = 0  # 当前偏差值
        self.running = True  # 运行状态标志

        # 调试参数
        self.debug_mode = True  # 显示调试窗口
        self.record_video = False  # 是否录制视频

        # 初始化视频录制（如果需要）
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (320, 240))

    def line_process(self, img):
        """优化的白底黑线处理方法"""
        # 1. 图像预处理
        frame = cv2.flip(img, -1)  # 根据实际需要决定是否翻转
        frame = frame[75:120, :]  # 裁剪ROI区域（可根据实际调整）

        # 2. 灰度转换与动态阈值处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊降噪

        # 动态计算阈值（比OTSU更稳定）
        mean_val = np.mean(gray)
        _, thresh = cv2.threshold(gray, max(mean_val - 40, 50), 255, cv2.THRESH_BINARY_INV)

        # 3. 形态学处理（针对白底黑线优化）
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 先闭运算填充小孔
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 再开运算去除噪声

        # 4. 轮廓检测与处理
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_center = frame.shape[1] // 2

        # 5. 黑线检测逻辑
        if len(contours) > 0:
            # 按面积排序，取前3个最大轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

            valid_contours = []
            for c in contours:
                # 过滤太小的轮廓
                if cv2.contourArea(c) < 50:
                    continue

                # 计算轮廓矩形特性
                rect = cv2.minAreaRect(c)
                width = min(rect[1])
                if width < 5:  # 过滤太窄的轮廓
                    continue

                valid_contours.append(c)

            if valid_contours:
                # 取最可能的主轮廓（最下方且较大的）
                c = max(valid_contours, key=lambda x: (cv2.boundingRect(x)[1], cv2.contourArea(x)))

                # 计算轮廓中心
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 跳动检测（改进版）
                    if len(self.history) >= 2:
                        last_cx = self.history[-1]
                        jump = abs(cx - last_cx)
                        avg_width = frame.shape[1] * 0.3  # 假设黑线约占30%宽度

                        if jump > avg_width * 0.5:  # 动态跳动阈值
                            self.error_flag = True
                            self.history = []

                    if not self.error_flag:
                        self.history.append(cx)
                        if len(self.history) > 5:
                            self.history.pop(0)

                        # 使用历史数据平滑
                        if len(self.history) > 2:
                            cx = int(np.mean(self.history[-3:]))

                        self.line_center = cx
                        self.deviation = frame_center - cx

                        # 调试绘制
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.line(frame, (frame_center, frame.shape[0]), (cx, cy), (255, 0, 0), 2)

                        # 显示实时数据
                        cv2.putText(frame, f"Dev: {self.deviation}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"Pos: {cx}", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                self.error_flag = True
        else:
            self.error_flag = True

        # 6. 错误状态处理
        if self.error_flag:
            self.deviation = 0
            cv2.putText(frame, "LINE LOST!", (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(self.history) > 0:
                # 尝试使用最后已知位置
                self.line_center = self.history[-1]
                self.deviation = frame_center - self.line_center

        # 返回处理后的图像和阈值图像用于调试
        return frame, thresh

    def control_movement(self):
        """根据偏差控制机器人运动"""
        if self.error_flag:
            # 丢失线路时的处理
            print("执行搜索模式...")
            # 这里添加搜索逻辑，例如原地旋转
        else:
            # PID控制或其他控制逻辑
            speed = 30  # 基础速度
            turn = int(self.deviation * 0.5)  # 转向系数

            print(f"控制命令: speed={speed}, turn={turn}")
            # 这里调用实际的运动控制接口

    def run(self):
        try:
            while self.running:
                start_time = time.time()

                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break

                # 处理图像
                debug_frame, thresh = self.line_process(frame)

                # 控制决策
                self.control_movement()

                # 显示结果
                if self.debug_mode:
                    cv2.imshow("Debug View", debug_frame)
                    cv2.imshow("Threshold", thresh)

                # 录制视频（如果需要）
                if self.record_video:
                    self.video_writer.write(debug_frame)

                # 计算帧率
                fps = 1.0 / (time.time() - start_time)
                print(f"FPS: {fps:.1f} | Deviation: {self.deviation}")

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode

        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.running = False
        self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("系统已关闭")


if __name__ == "__main__":
    # 创建并启动跟踪器
    follower = LineFollower()

    # 可选：等待摄像头预热
    time.sleep(2.0)

    # 运行主循环
    follower.run()