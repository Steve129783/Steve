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

# 1. 初始化模型并加载权重
def load_model(weight_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None
    ).cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

# 2. 与训练一致的预处理流程
def get_infer_transform():
    return A.Compose([
        A.Resize(height=364, width=388),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# 3. 推理并返回 mainbody 掩膜
def predict_mainbody(model, image_path):
    image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    transform = get_infer_transform()
    image_tensor = transform(image=image)["image"].unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(image_tensor)             # [1, 2, H, W]
        body_logits = output[0, 0]               # 通道0：mainbody
        body_prob = torch.sigmoid(body_logits).cpu().numpy()
        body_mask = (body_prob > 0.8).astype(np.uint8) * 255
        return body_mask

# 4. 可视化
def show_mask(mask):
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mainbody Mask")
    plt.axis("off")
    plt.show()

# 5. 保存为 Labelme JSON
def save_mask_as_labelme_json(mask, image_path, save_path):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) != 2:
            continue
        points = contour.tolist()
        shapes.append({
            "label": "main body",  # ❗ 改为 main body
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

    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)

# 6. 批量推理并保存 JSON
def batch_predict_mainbody_to_json(model, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tif_list = glob.glob(os.path.join(input_folder, "*.tif"))
    png_list = glob.glob(os.path.join(input_folder, "*.png"))
    image_paths = sorted(tif_list + png_list)

    for image_path in image_paths:
        print(f"[INFO] Processing: {image_path}")
        try:
            original_image = cv2.imread(image_path)
            orig_h, orig_w = original_image.shape[:2]

            mainbody_mask = predict_mainbody(model, image_path)
            restored_mask = cv2.resize(mainbody_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            filename = os.path.splitext(os.path.basename(image_path))[0] + "_mainbody.json"
            save_path = os.path.join(output_folder, filename)

            save_mask_as_labelme_json(restored_mask, image_path, save_path)
            print(f"[✅] Saved: {save_path}")

        except Exception as e:
            print(f"[❌] Failed to process {image_path}: {e}")

# 7. 使用模型进行批量推理
model_path = r"D:\Study\Postgraduate\S2\Project\Code\Resource\semantic_cache\1_2_3_4\1\Binary\mainbody\2B_mainbody.pth"
input_folder = r"E:\Project_VAE\256_slice_padding\1_2_3_4\1"
output_folder = r"E:\Project_VAE\Test\1_2_3_4\1\MB"

model = load_model(model_path)
batch_predict_mainbody_to_json(model, input_folder, output_folder)
