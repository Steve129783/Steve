import torch
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import glob
import base64

# 1. 初始化单通道模型并加载权重

def load_model(weight_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

# 2. 与训练一致的预处理流程

def get_infer_transform():
    return A.Compose([
        A.Resize(height=319, width=442),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# 3. 推理并返回主躯干二值掩膜

def predict_mainbody(model, image_path, threshold=0.5):
    img_bgr = cv2.imread(image_path)
    image = img_bgr[:, :, ::-1]
    h0, w0 = image.shape[:2]

    transform = get_infer_transform()
    image_tensor = transform(image=image)["image"].unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(image_tensor)      # [1, 1, H, W]
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        # 二值化
        mask = (prob > threshold).astype(np.uint8) * 255
        # 恢复到原始尺寸
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return mask

# 4. 保存为 LabelMe JSON，仅保留最大面积的轮廓

def save_mask_as_labelme_json(mask, image_path, save_path, label_name='main body'):
    # 连通性处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[⚠️] No contour found for {image_path}")
        return

    # 选最大轮廓
    c = max(contours, key=cv2.contourArea).squeeze()
    if c.ndim != 2:
        return

    points = c.tolist()
    shapes = [{
        "label": label_name,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }]

    # 读取图片并编码
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    h, w = mask.shape
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": img_b64,
        "imageHeight": h,
        "imageWidth": w
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 5. 批量推理并保存

def batch_predict(input_folder, output_folder, model_path):
    model = load_model(model_path)
    os.makedirs(output_folder, exist_ok=True)
    exts = ['*.png','*.jpg','*.jpeg','*.tif','*.tiff']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    image_paths = sorted(image_paths)

    for img_p in image_paths:
        print(f"Processing {img_p}...")
        mask = predict_mainbody(model, img_p)
        base = os.path.splitext(os.path.basename(img_p))[0]
        save_p = os.path.join(output_folder, base + '.json')
        save_mask_as_labelme_json(mask, img_p, save_p)
        print(f"Saved {save_p}")

# 6. 运行示例
if __name__ == '__main__':
    file_name = '17_18_19_20'
    gn = '20'
    model_path = rf'E:\Project_SNV\0S\s1_cache\{file_name}\best_model.pth'
    in_dir = rf'D:\Study\Postgraduate\S2\Project\Code\Resource\Original\{file_name}\{gn}'
    out_dir = rf'E:\Project_SNV\0S\2_MB\{gn}'
    batch_predict(in_dir, out_dir, model_path)
