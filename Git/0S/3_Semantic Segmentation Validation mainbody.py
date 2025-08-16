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

# 1. Initialize a single-channel segmentation model and load weights

def load_model(weight_path):
    model = smp.Unet(
        encoder_name="resnet34",           # Use ResNet34 encoder
        encoder_weights="imagenet",       # Pretrained on ImageNet
        in_channels=3,                     # Three input channels (RGB)
        classes=1,                         # Single output class
        activation=None                    # No activation (logits output)
    ).cuda()
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

# 2. Define preprocessing transformations consistent with training

def get_infer_transform():
    return A.Compose([
        A.Resize(height=319, width=442),
        A.Normalize(mean=(0.5,0.5,0.5),
                    std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])

# 3. Perform inference and return a binary mask of the main body

def predict_mainbody(model, image_path, threshold=0.5):
    # Read image in BGR format then convert to RGB
    img_bgr = cv2.imread(image_path)
    image = img_bgr[:, :, ::-1]
    h0, w0 = image.shape[:2]  # Original dimensions

    # Apply preprocessing
    transform = get_infer_transform()
    image_tensor = transform(image=image)["image"].unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(image_tensor)         # Model output logits [1, 1, H, W]
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        # Threshold to create a binary mask
        mask = (prob > threshold).astype(np.uint8) * 255
        # Resize mask back to original image size
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return mask

# 4. Save the mask as a LabelMe-format JSON, keeping only the largest contour

def save_mask_as_labelme_json(mask, image_path, save_path, label_name='main body'):
    # Morphological closing then opening to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[⚠️] No contour found for {image_path}")
        return

    # Select the largest contour by area
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

    # Read and encode image data in base64
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
    # Write JSON to disk
    with open(save_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 5. Batch process images: run inference and save results

def batch_predict(input_folder, output_folder, model_path):
    # Load the pretrained model
    model = load_model(model_path)
    os.makedirs(output_folder, exist_ok=True)
    # Gather all image file paths with supported extensions
    exts = ['*.png','*.jpg','*.jpeg','*.tif','*.tiff']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    image_paths = sorted(image_paths)

    # Process each image
    for img_p in image_paths:
        print(f"Processing {img_p}...")
        mask = predict_mainbody(model, img_p)
        base = os.path.splitext(os.path.basename(img_p))[0]
        save_p = os.path.join(output_folder, base + '.json')
        save_mask_as_labelme_json(mask, img_p, save_p)
        print(f"Saved {save_p}")

# 6. Example execution
if __name__ == '__main__':
    file_name = '17_18_19_20'
    gn = '20'
    model_path = rf'E:\Project_SNV\0S\s1_cache\{file_name}\best_model.pth'
    in_dir = rf'D:\Study\Postgraduate\S2\Project\Code\Resource\Original\{file_name}\{gn}'
    out_dir = rf'E:\Project_SNV\0S\2_MB\{gn}'
    batch_predict(in_dir, out_dir, model_path)
