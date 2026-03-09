import cv2

def check_cameras(max_test=5):
    """检查可用的摄像头索引"""
    available_cams = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 在Windows上使用DSHOW
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

if __name__ == "__main__":
    cameras = check_cameras()
    print(f"可用的摄像头索引: {cameras}")