import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from CNN import ImageModel
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_like_mnist(image_path):
    # 1. 读图，转灰度
    img = Image.open(image_path).convert('L')
    
    # 2. 确保是黑底白字 (如果你的原图是白底黑字，用 ImageOps.invert 反色)
    # 你的图已经是黑底白字了，不用反色，这里确认一下
    # img = ImageOps.invert(img) 
    
    # 3. 二值化（去除边缘的灰色过渡，让笔画变实心）
    # 设定一个阈值（比如 50），大于它的全变成纯白 255，小于它的变纯黑 0
    img_array = np.array(img)
    img_array = (img_array > 50) * 255
    img_array = img_array.astype(np.uint8)
    
    # 4. 寻找数字的边界框（Bounding Box）
    coords = np.argwhere(img_array > 0)
    if len(coords) == 0:
        print("警告：图片全黑，没有找到数字！")
        return transforms.ToTensor()(Image.new('L', (28, 28), 0))
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 5. 把数字裁剪出来
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]
    cropped_img = Image.fromarray(cropped)
    
    # 6. 等比例缩放（MNIST的习惯：最长边缩放到 20 像素，保留比例）
    w, h = cropped_img.size
    if w > h:
        new_w = 20
        new_h = int(h * 20 / w)
    else:
        new_h = 20
        new_w = int(w * 20 / h)
        
    # 防止尺寸缩成 0
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # 使用 LANZOS 算法缩放，能最大程度保留笔画边缘的平滑度
    cropped_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 7. 放到 28x28 的画布正中间
    canvas = Image.new('L', (28, 28), 0) # 创建纯黑画布
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(cropped_img, (paste_x, paste_y))
    
    # 8. 转 Tensor
    return transforms.ToTensor()(canvas)

def predict_single(image_path, model, device):
    # 调用上面的终极预处理
    tensor_img = preprocess_like_mnist(image_path) 

    # ================= 新增：可视化代码 =================
    # 把 Tensor 从 [1, 28, 28] 变回 [28, 28]
    img_to_show = tensor_img.squeeze().numpy()
    
    plt.figure(figsize=(3, 3))
    # cmap='gray' 表示用灰度图显示
    plt.imshow(img_to_show, cmap='gray')
    plt.title(f"预处理后的图: {image_path}")
    plt.axis('off') # 关掉坐标轴
    plt.show()
    # ====================================================
    
    # 增加 Batch 维度 [1, 28, 28] -> [1, 1, 28, 28]
    tensor_img = tensor_img.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs[0, pred].item()
        
    print(f"图片 {image_path} -> 预测结果: {pred}, 把握: {conf*100:.2f}%")

# 你的测试代码...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageModel().to(device)
model.load_state_dict(torch.load('./model/digital_image_model.pth'))

predict_single(r'data\digital\1.png', model, device)
predict_single(r'data\digital\2.png', model, device)
predict_single(r'data\digital\7.png', model, device)
