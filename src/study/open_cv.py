import cv2
import numpy as np

def LEAFS():
    img = cv2.imread(r"C:\Users\Interstellar\Desktop\ROBOCON_study\robocon-vision\data\samples\low_contrast_leaf.png")
    if img is None:
        raise SystemExit("图片没读到，检查路径")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 先去椒盐噪声（中值，小核）
    denoised = cv2.medianBlur(gray, 3)

    # 2. 轻高斯，压掉残余颗粒
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

    # 3. CLAHE 增强对比度（clipLimit 调小，避免放大噪声）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. 再轻高斯 + Canny
    blur2 = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur2, 50, 150)

    # 5. 形态学闭运算连接断边（用你已有的 morphologyEx）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数:", len(contours))
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

if __name__ == "__main__":
    LEAFS()
