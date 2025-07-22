from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from copy import deepcopy
import base64
# Automated identify the defects from original images by OpenCV and input to labelme for marking
import cv2

# Read the image Serial Number from file
def extract_last_number(filename):
    nums = re.findall(r'\d+', filename)
    if not nums:
        return None
    
    return (nums[-1])

num_list = []

image_dir = r'D:\Study\Postgraduate\S2\Project\Code\Resource\Original\1_2_3_4\1'
for fname in os.listdir(image_dir):
    if not fname.lower().endswith('.tif'):
        continue

    last_num = extract_last_number(fname)
    num_list.append(last_num)

file_names = []

for i in range(len(num_list)):
    fname = f'crop_Componenets_1.transformed{num_list[i]}.tif'
    full_path = os.path.join(image_dir, fname)
    file_names.append(full_path)
    


def load_template(template_json_path: Path, template_image_path: Path, target_label='main body'):
    """
    Read JSON, images, then extract the 'main body' polygon and template graydiagram & patch
    """
    # 1. Read template JSON
    tmpl = json.loads(template_json_path.read_text(encoding='utf-8'))
    shapes = [s for s in tmpl.get('shapes', []) if s.get('label') == target_label]
    
    if not shapes:
        raise ValueError(f"No shapes with label '{target_label}' in {template_json_path}")
    template_shape = shapes[0]

    # 2. Read template graydiagram
    tmpl_img = cv2.imread(str(template_image_path), cv2.IMREAD_GRAYSCALE)
    if tmpl_img is None:
        raise FileNotFoundError(f"Cannot read image {template_image_path}")

    # 3. Generate patch from polygon（mask & bitwise_and）
    poly_pts = np.array(template_shape['points'], dtype=np.int32)
    mask = np.zeros_like(tmpl_img, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    patch = cv2.bitwise_and(tmpl_img, tmpl_img, mask=mask)

    # 4. extract JSON public field（excluding imageData, imagePath, shapes, size）
    common = {
        k: tmpl[k] for k in tmpl.keys()
        if k not in ('imageData', 'imagePath', 'shapes', 'imageHeight', 'imageWidth')
    }

    return common, template_shape, tmpl_img, patch

def encode_image_to_b64(img_path: Path) -> str:
    """encode the images to Base64 String。"""
    return base64.b64encode(img_path.read_bytes()).decode('utf-8')

def estimate_translation_ecc(tmpl_f: np.ndarray, img_f: np.ndarray):
    """
    Use ECC algorithm estimate the Translation，input & ouput images should be float32 and normalized to [0,1]
    Return (dx, dy)
    """
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

    cc, warp_matrix = cv2.findTransformECC(
        tmpl_f, img_f, warp_matrix,
        warp_mode,
        criteria,
        inputMask=None,
        gaussFiltSize=5
    )
    dx, dy = warp_matrix[0,2], warp_matrix[1,2]
    return dx, dy

def make_json_for_image_ecc(common: dict,
                            template_shape: dict,
                            tmpl_img: np.ndarray,
                            tmpl_patch: np.ndarray,
                            img_path: Path):
    """
    Focus on single image：
    1) Read grayscale & normalization
    2) ECC alignment and estimate translation
    3) translate the polygon
    4) Generate new JSON（Including recoding imageData）
    """
    # 1. Read grayscale & normalization
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot read image {img_path}")
    tmpl_f = tmpl_img.astype(np.float32) / 255.0
    img_f  = img_gray.astype(np.float32) / 255.0

    # 2. ECC alignment and estimate translation
    dx, dy = estimate_translation_ecc(tmpl_f, img_f)

    # 3. translate the polygon
    orig_pts = np.array(template_shape['points'], dtype=np.float32)
    shifted  = (orig_pts + np.array([dx, dy])).tolist()

    # 4. Generate new JSON（Including recoding imageData）
    j = dict(common)
    j['imagePath']   = img_path.name
    j['imageData']   = encode_image_to_b64(img_path)
    with Image.open(img_path) as im:
        j['imageHeight'], j['imageWidth'] = im.height, im.width

    new_shape = deepcopy(template_shape)
    new_shape['points'] = shifted
    j['shapes'] = [new_shape]
    return j

def batch_apply_with_ecc(template_json: Path,
                         template_image: Path,
                         images_dir: Path,
                         output_dir: Path,
                         target_label='main body'):
    """
    Implement templete ECC alignment to offset images in images_dir，and generate the corresponding LabelMe JSON to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    common, tmpl_shape, tmpl_img, tmpl_patch = load_template(
        template_json, template_image, target_label
    )

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            continue
        try:
            new_json = make_json_for_image_ecc(
                common, tmpl_shape, tmpl_img, tmpl_patch, img_path
            )
            out_path = output_dir / f"{img_path.stem}.json"
            out_path.write_text(
                json.dumps(new_json, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            print(f"✓ Generated {out_path.name}")
        except Exception as e:
            print(f"✗ Failed {img_path.name}: {e}")

if __name__ == '__main__':
    template_json_path  = Path(r"D:\Study\Postgraduate\S2\Project\Code\Resource\mainbody label\13_14_15_16\14\crop_Components_14.transformed151.json") # Mask position
    template_image_path = Path(r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\13_14_15_16\14\crop_Components_14.transformed151.tif")
    images_dir          = Path(r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\13_14_15_16\14") # image file
    output_dir          = Path(r"E:\Project_SNV\0S\1_Full_Raw_MB\14") # main body file

    batch_apply_with_ecc(
        template_json  = template_json_path,
        template_image = template_image_path,
        images_dir     = images_dir,
        output_dir     = output_dir,
        target_label   = 'main body'
    )


