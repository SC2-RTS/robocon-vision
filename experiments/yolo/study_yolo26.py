from ultralytics import YOLO

# 加载高效的 YOLO26 Nano 模型
model = YOLO("yolo26n.pt")

def quick_start():
    # 在标准 COCO8 数据集上训练
    # 自动应用默认超参数和数据增强
    results = model.train(
        data="coco8.yaml",
        epochs=100,
        imgsz=640,
        device="0",  # 第一块 CUDA GPU；使用 device="cpu" 在 CPU 上训练
    )

    # 对源图像运行端到端无 NMS 推理

    predictions = model("https://ultralytics.com/images/bus.jpg", nms=False)
    # 可视化检测结果
    predictions[0].show()

def detect_oneself():
    # 在本地摄像头上运行实时检测
    results = model.predict(source=0, show=True, save=False, device="0", conf=0.25)  # 设备 0 是第一块 CUDA GPU
    results[0].show()

def model_training():
    results = model.train(data="datasets\leaves\data.yaml", epochs=50, device="0", name='yolo26n-leaves')

if __name__ == "__main__":
    # quick_start()
    # detect_oneself()
    model_training()
