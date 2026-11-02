from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt

SRC = "data/samples/wobble.mp4"
DST = "data/out/my_yolo_manual.mp4"
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(SRC)
if cap.isOpened() == False:
    print("Error opening video stream or file")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

delpy = int(1000 / fps)
writer = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
index = 0
n_box = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print('视频播放结束')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    # hist = cv2.calcHist([median], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.xlabel("pixel value")
    # plt.ylabel("count")
    # plt.show()
    ret, bw = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print("Otsu 选的阈值:", ret)   # 47
    result = model(bw, agnostic_nms=True, verbose=False, conf=0.20)[0]

    index += 1
    for x1, y1, x2, y2, conf, cls in result.boxes.data.cpu().numpy():
        label = f"{result.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # 给文字加底色
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1) - th - 6),
                      (int(x1) + tw + 4, int(y1)), (0, 0, 255), -1)
        cv2.putText(frame, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        n_box += 1

    cv2.putText(frame, f"frame {index}  boxes {n_box}  conf>={0.20}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    writer.write(frame)
    cv2.namedWindow('my manual draw', cv2.WINDOW_NORMAL)
    cv2.imshow('my manual draw', frame)
    if cv2.waitKey(delpy) & 0xFF in (27, ord('q')):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"完成：{DST}  帧数 {index}  总框数 {n_box}")
