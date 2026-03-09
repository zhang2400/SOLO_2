# from ultralytics import YOLO
#
# # Load a YOLO11n PyTorch model
# model = YOLO("runs/detect/train7/weights/best.pt")
#
# # Export the model to NCNN format
# model.export(format="ncnn",data="data.yaml",half = True)  # creates 'yolo11n_ncnn_model'

from ultralytics import YOLO

# 加载你训练好的模型（假设保存为best.pt）
model = YOLO('runs/detect/train7/weights/best.pt')

# 导出为ONNX格式
success = model.export(format='onnx', imgsz=320, simplify=True, opset=12)
print(f'导出ONNX {"成功" if success else "失败"}')






