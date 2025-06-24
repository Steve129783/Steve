import torch
import json
import os
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import glob
import base64

# 1. 初始化模型并加载权重
def load_model_only(weight_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None
    ).cuda()

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# 2. 定义与训练一致的预处理流程
def get_infer_transform():
    return A.ReplayCompose([
        A.Resize(height=354, width=348 ),
        # 这里如果你还有旋转等 augment 也要写进去（用于训练时一致）
        # A.RandomRotate90(p=1.0), 例如训练用了的话
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# 3. 执行推理并返回 defect 掩膜
def predict_defect(model, image_path):
    image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    transform = get_infer_transform()
    image_tensor = transform(image=image)["image"].unsqueeze(0).cuda()  # [1, 3, H, W]

    with torch.no_grad():
        output = model(image_tensor)             # [1, 2, H, W]
        defect_logits = output[0, 1]             # 第1通道：defect
        defect_prob = torch.sigmoid(defect_logits).cpu().numpy()
        defect_mask = (defect_prob > 0.1).astype(np.uint8) * 255  # 二值图
        return defect_mask

# 4. 可视化结果
def show_mask(mask):
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Defect Mask")
    plt.axis("off")
    plt.show()

# 5. MASK generator
def save_mask_as_labelme_json(mask, image_path, save_path):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) != 2:
            continue
        points = contour.tolist()
        shapes.append({
            "label": "defect",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    h, w = mask.shape

    # 读取图片二进制数据，转base64字符串
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    json_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": img_b64,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=4)

def batch_predict_defect_to_json(model, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tif_list = glob.glob(os.path.join(input_folder, "*.tif"))
    png_list = glob.glob(os.path.join(input_folder, "*.png"))
    image_paths = sorted(tif_list + png_list)

    for image_path in image_paths:
        print(f"[INFO] Processing: {image_path}")
        try:
            # 读取图像
            original_image = cv2.imread(image_path)
            orig_h, orig_w = original_image.shape[:2]

            # 推理
            defect_mask = predict_defect(model, image_path)

            # resize 回原图尺寸
            restored_mask = cv2.resize(defect_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # 构建保存路径
            filename = os.path.splitext(os.path.basename(image_path))[0] + "_defect.json"
            save_path = os.path.join(output_folder, filename)

            # 保存 JSON
            save_mask_as_labelme_json(restored_mask, image_path, save_path)
            print(f"[✅] Saved: {save_path}")
        
        except Exception as e:
            print(f"[❌] Failed to process {image_path}: {e}")

model_path = r"D:\Study\Postgraduate\S2\Project\Code\Resource\semantic_cache\1_2_3_4\1\Binary\1checkpoint\2B_defect.pth"
input_folder = r"E:\Project_VAE\256_slice_padding\1_2_3_4\1"
output_folder = r"E:\Project_VAE\Test\1_2_3_4\1\DF"

model = load_model_only(model_path)
batch_predict_defect_to_json(model, input_folder, output_folder)


