import cv2

# 初始化摄像头（Windows 建议添加 CAP_DSHOW 参数）
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 设置视频参数（分辨率、帧率可自定义）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # 帧率与摄像头实际性能相关，过高可能无效

# 定义视频编码格式（MP4 常用 'mp4v' 或 'X264'）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output1.mp4', fourcc, fps, (frame_width, frame_height))

# 检查摄像头和写入器是否正常启动
if not cap.isOpened() or not out.isOpened():
    print("初始化失败: 摄像头或视频写入器异常")
    exit()

# 实时录制循环
while True:
    ret, frame = cap.read()
    if not ret:
        print("帧获取中断")
        break

    out.write(frame)  # 写入当前帧到视频文件
    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
# import cv2
#
# # 打开默认摄像头（索引0）
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # CAP_DSHOW 可解决部分系统兼容性问题:ml-citation{ref="6" data="citationList"}
#
# # 检查摄像头是否成功打开
# if not cap.isOpened():
#     print("摄像头打开失败")
#     exit()
#
# # 实时读取帧并显示
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("无法获取帧")
#         break
#
#     cv2.imshow('Camera Preview', frame)
#
#     # 按 'q' 键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()
