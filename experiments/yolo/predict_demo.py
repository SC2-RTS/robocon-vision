from ultralytics import YOLO

# bus.jpg
# # model = YOLO("yolo26n.pt")  # 加载预训练的 YOLO26n 检测模型
# # results = model("C:\\Users\\Interstellar\\Desktop\\ROBOCON_study\\robocon-vision\\experiments\\yolo\\bus.jpg", save=True)  # 进行预测并保存标注后的图像

# results = YOLO("yolov8n.pt")('C:\\Users\\Interstellar\\Desktop\\ROBOCON_study\\robocon-vision\\experiments\\yolo\\bus.jpg', save=True)  # 进行预测并保存标注后的图像
# print('r[0].boxes.xyxy: ', results[0].boxes.xyxy)
# print('r[0].boxesconf: ', results[0].boxes.conf)
# print('r[0].boxes.cls: ', results[0].boxes.cls)
# print(results[0].names[int(results[0].boxes.cls[0])])     # → 'bus'

# wobble.mp4
model = YOLO("yolov8n.pt", task='detect')  # 加载预训练的 YOLOv8n 检测模型
results = model(source=r'C:\Users\Interstellar\Desktop\ROBOCON_study\robocon-vision\data\samples\wobble.mp4', save=True, conf=0.1)  # 进行预测并保存标注后的视频
