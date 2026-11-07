import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

def main():
    """用 [1,300,6] 的参考答案自检 [1,5,8400] 的解码 + NMS。"""
    sess = ort.InferenceSession(
        "runs/detect/yolo26n-leaves/weights/yolo26_1x300x6.onnx",
        providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name            # 'images'

    img = cv2.imread("datasets/leaves/images/val/frame_0120.jpg")
    h, w = img.shape[:2]
    r = min(640 / h, 640 / w)                                   # 1. letterbox 缩放比
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((640, 640, 3), 114, np.uint8)              # 2. 灰边填充
    dx, dy = (640 - nw) // 2, (640 - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))

    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {inp: x})[0][0]                        # (300, 6)

    for x1, y1, x2, y2, conf, cls in out[out[:, 4] > 0.5]:
        print(f"conf={conf:.3f} cls={int(cls)} "
            f"({(x1-dx)/r:.0f},{(y1-dy)/r:.0f})-({(x2-dx)/r:.0f},{(y2-dy)/r:.0f})")

def cross_validation_testing():
    m = YOLO("runs/detect/yolo26n-leaves/weights/best.pt")
    r = m("datasets/leaves/images/val/frame_0120.jpg", conf=0.5, verbose=False)[0]
    print(r.boxes.xyxy.cpu().numpy())

if __name__ == "__main__":
    # main()
    cross_validation_testing()
