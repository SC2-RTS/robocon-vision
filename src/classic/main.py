from detect_targets import Params, find_targets
import cv2

cap = cv2.VideoCapture(r"C:\Users\Interstellar\Desktop\ROBOCON_study\robocon-vision\data\samples\wobble.mp4")                 # SOURCE 可以是 "xxx.mp4" 或 0（摄像头）
fps = cap.get(cv2.CAP_PROP_FPS)
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

delay = int(1000 / fps) if fps > 0 else 33
idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break                           # 坑1：视频结束，也是摄像头断开

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    targets, mask = find_targets(gray, Params())   # 坑2：不是 detect()

    idx += 1
    for t in targets:    #画红框 + 中心点 + FPS
        cv2.putText(frame, "%d/%d" % (idx, total), (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(max(1, delay)) & 0xFF in (27, ord("q")): break    # 坑3：必须 & 0xFF
