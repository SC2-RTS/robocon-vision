from ultralytics import YOLO


def my_box_train():
    model = YOLO('yolov8n.pt')
    results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=50, device=0, name='train_my_box', seed=0)

def my_box_train_pt1():
    model = YOLO('yolov8n.pt')
    # results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=40, batch=8, patience=20, warmup_epochs=1, mosaic=0.0, degrees=5, scale=0.3, device=0, name='train_my_box_pt1', seed=0)
    results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=50, device=0, imgsz=1080, name='my_noai_train_my_box', seed=0)

def my_box_train_pt2():
    model = YOLO('yolov8n.pt')
    # results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=40, batch=8, patience=20, warmup_epochs=1, mosaic=0.0, degrees=5, scale=0.3, device=0, name='train_my_box_pt1', seed=0)
    results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=50, device=0, imgsz=1080, batch=8, name='my_noai_train_my_box2', seed=0)

def my_box_train_pt2_add_dataset():
    model = YOLO('yolov8n.pt')
    # results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=40, batch=8, patience=20, warmup_epochs=1, mosaic=0.0, degrees=5, scale=0.3, device=0, name='train_my_box_pt1', seed=0)
    results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=50, device=0, imgsz=1080, batch=8, name='my_noai_train_my_box2_add_dataset', seed=0)

def my_box_pred():
    model = YOLO(r'runs\detect\train_my_box\weights\best.pt')
    results = model.predict(source=r'datasets\redmi_buds5_box\images\val', conf=0.45, iou=0.45, save=True, save_conf=True, save_txt=True, name='pred_my_box')

def my_box_pred_pt1():
    model = YOLO(r'runs\detect\train_my_box_pt1\weights\best.pt')
    results = model.predict(source=r'datasets\redmi_buds5_box\images\val', conf=0.6, iou=0.45, save=True, save_conf=True, save_txt=True, name='pred_my_box_pt1')

def my_noai_box_pred_pt1():
    model = YOLO(r'runs\detect\my_noai_train_my_box\weights\best.pt')
    results = model.predict(source=r'datasets\redmi_buds5_box\images\val', conf=0.7, iou=0.45, save=True, save_conf=True, save_txt=True, name='pred_my_noai_box_pt1')

def my_noai_box_pred_pt2():
    model = YOLO(r'runs\detect\my_noai_train_my_box2\weights\best.pt')
    results = model.predict(source=r'datasets\redmi_buds5_box\images\val', conf=0.679, iou=0.45, save=True, save_conf=True, save_txt=True, name='pred_my_noai_box_pt2')

def my_noai_box_pred_pt2_add_dataset():
    model = YOLO(r'runs\detect\my_noai_train_my_box2_add_dataset\weights\best.pt')
    results = model.predict(source=r'datasets\redmi_buds5_box\images\val', conf=0.7, iou=0.45, save=True, save_conf=True, save_txt=True, imgsz=1500, name='pred_my_noai_box_pt2_add_dataset')


def my_train():
    model = YOLO('yolov8n.pt')
    results = model.train(data=r'datasets\redmi_buds5_box\data.yaml', epochs=100, device=0, name='my_train', seed=0)

def my_pred():
    model = YOLO(r'runs\detect\my_train\weights\best.pt')
    results = model.predict(source=r'datasets\redmi_buds5_box\images\val', conf=0.753, name='my_pred', save=True, save_conf=True, save_txt=True)

def my_pred_test():
    model = YOLO(r'runs\detect\my_train\weights\best.pt')
    results = model.predict(source=r'datasets\test', conf=0.753, name='my_pred_test', save=True, save_conf=True, save_txt=True)


if __name__ == '__main__':
    # my_box_train_pt1()
    my_pred_test()

