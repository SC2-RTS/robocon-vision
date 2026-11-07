from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")  # 加载官方模型
# model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

# 导出模型
model.export(format="onnx")
