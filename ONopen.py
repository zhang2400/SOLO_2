import cv2

# 初始化摄像头（兼容Windows平台可添加cv2.CAP_DSHOW参数）
cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头:ml-citation{ref="1,2" data="citationList"}

if not cap.isOpened():  # 检测摄像头初始化状态:ml-citation{ref="3" data="citationList"}
    print("Error: 无法打开摄像头")
    exit()

# 设置摄像头参数（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  # 高度:ml-citation{ref="3,7" data="citationList"}

try:
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("Error: 无法获取视频帧")
            break

        # 显示实时画面
        cv2.imshow('Camera Feed', frame)

        # 按Q键退出（ASCII码转换）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
