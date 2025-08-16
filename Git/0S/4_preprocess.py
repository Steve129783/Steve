import os
import json
import base64
import csv
import numpy as np
from io import BytesIO
from PIL import Image
from skimage.transform import AffineTransform, warp
from shapely.geometry import Polygon


def compute_consistent_orientation(points):
    """
    Compute the rotation angle (radians) and centroid required to align 
    the longest edge of a polygon horizontally (within ±45°).
    Returns the angle (in radians) and the centroid of the polygon.
    """
    # Convert list of points to NumPy array and create Shapely polygon
    pts = np.array(points, dtype=float)
    poly = Polygon(pts)
    # Compute minimum-area rotated rectangle
    min_rect = poly.minimum_rotated_rectangle
    # Get rectangle vertices (remove the duplicate closing point)
    rect = np.array(min_rect.exterior.coords)[:-1]

    # Construct rectangle edges and their lengths
    edges = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]
    lengths = [np.linalg.norm(p2 - p1) for p1, p2 in edges]
    # Identify the longest edge
    idx = int(np.argmax(lengths))
    p1, p2 = edges[idx]

    # Compute edge angle
    raw = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    # Normalize to (-pi/2, pi/2]
    if raw > np.pi / 2:
        raw -= np.pi
    elif raw <= -np.pi / 2:
        raw += np.pi
    # If closer to vertical, rotate by ±90°
    if raw > np.pi / 4:
        angle = raw - np.pi / 2
    elif raw < -np.pi / 4:
        angle = raw + np.pi / 2
    else:
        angle = raw

    # Rotation center is the centroid of the minimum rectangle
    center = np.array(min_rect.centroid.coords[0])
    return angle, center


def rotate_and_center(image, shapes, angle, center, output_size=400):
    """
    Rotate the image and its corresponding shape points around the given center
    by the specified angle, then move the main body center to the center 
    of a fixed-size canvas.

    Returns the transformed image and the updated shapes.
    """
    cx, cy = center
    # Build affine transform: translate → rotate → translate back
    T = (
        AffineTransform(translation=(-cx, -cy))
        + AffineTransform(rotation=-angle)
        + AffineTransform(translation=(cx, cy))
    )

    # Apply affine transformation to the image
    if image.ndim == 2:
        img_rot = warp(image, T.inverse, order=1, preserve_range=True)
    else:
        chans = [
            warp(image[..., c], T.inverse, order=1, preserve_range=True)
            for c in range(image.shape[2])
        ]
        img_rot = np.stack(chans, axis=-1)
    img_rot = img_rot.astype(image.dtype)

    # Apply transformation to all shape points
    shapes_rot = []
    for shp in shapes:
        pts = np.array(shp['points'], dtype=float)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # homogeneous coords
        pts_new = (T.params @ pts_h.T).T[:, :2]
        new_shape = shp.copy()
        new_shape['points'] = pts_new.tolist()
        shapes_rot.append(new_shape)

    # Create fixed-size canvas
    canvas = np.zeros(
        (output_size, output_size) + (() if image.ndim == 2 else (image.shape[2],)),
        dtype=image.dtype
    )
    # Find "main body" shape and compute its centroid
    main_body = next(s for s in shapes_rot if s.get('label') == 'main body')
    mb_center = np.array(main_body['points'], dtype=float).mean(axis=0)
    dx = int(round(output_size / 2 - mb_center[0]))
    dy = int(round(output_size / 2 - mb_center[1]))

    # Compute placement area on the canvas
    x0, y0 = dx, dy
    x1, y1 = x0 + img_rot.shape[1], y0 + img_rot.shape[0]
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(output_size, x1), min(output_size, y1)
    sx0, sy0 = xs0 - x0, ys0 - y0
    sx1, sy1 = sx0 + (xs1 - xs0), sy0 + (ys1 - ys0)

    # Paste rotated image onto canvas
    canvas[ys0:ys1, xs0:xs1] = img_rot[sy0:sy1, sx0:sx1]

    # Update shape points to canvas coordinates
    final_shapes = []
    for shp in shapes_rot:
        pts = np.array(shp['points'], dtype=float)
        pts[:, 0] += x0
        pts[:, 1] += y0
        new_shape = shp.copy()
        new_shape['points'] = pts.tolist()
        final_shapes.append(new_shape)

    return canvas, final_shapes


def encode_image_to_base64(arr, fmt="PNG"):
    """
    Encode a NumPy image array into a Base64 string with the given format.
    """
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == "__main__":
    # Batch processing config
    file_name = '17_18_19_20'
    N = 20
    img_dir    = fr"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\{file_name}\{N}"
    json_dir   = fr"E:\Project_SNV\0S\2_MB\{N}"
    out_img_d  = fr"E:\Project_SNV\0S\3_full_img\{N}"
    out_json_d = fr"E:\Project_SNV\0S\4_moved_MB\{N}"

    # Create output directories
    os.makedirs(out_img_d, exist_ok=True)
    os.makedirs(out_json_d, exist_ok=True)

    # Create CSV file to record rotation angles
    csv_path = os.path.join(out_json_d, "rotation_angles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["index", "angle_degrees"])

        idx = 0
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(".tif"):
                continue

            base = os.path.splitext(fn)[0]
            img_p  = os.path.join(img_dir, fn)
            json_p = os.path.join(json_dir, base + ".json")
            # Skip if JSON does not exist
            if not os.path.exists(json_p):
                print(f"Skipping {base}, JSON not found.")
                continue

            # Load image and JSON
            img = np.array(Image.open(img_p))
            with open(json_p, "r", encoding="utf-8") as f:
                data = json.load(f)

            shapes = data.get("shapes", [])
            main_body = next((s for s in shapes if s.get("label") == "main body"), None)
            if main_body is None:
                print(f"Skipping {base}, 'main body' label missing.")
                continue

            # Compute rotation angle and center
            angle, center = compute_consistent_orientation(main_body["points"])
            deg = np.degrees(angle)
            writer.writerow([idx, f"{deg:.2f}"])

            # Rotate and center
            img_out, shapes_out = rotate_and_center(img, shapes, angle, center, output_size=400)

            # Save aligned image
            out_img = os.path.join(out_img_d, f"{idx}.png")
            Image.fromarray(img_out).save(out_img)

            # Update JSON fields
            data["shapes"]      = shapes_out
            data["imageData"]   = encode_image_to_base64(img_out, fmt="PNG")
            data["imagePath"]   = os.path.basename(out_img)
            data["imageHeight"] = img_out.shape[0]
            data["imageWidth"]  = img_out.shape[1]

            out_json = os.path.join(out_json_d, f"{idx}.json")
            with open(out_json, "w", encoding="utf-8") as fw:
                json.dump(data, fw, ensure_ascii=False, indent=2)

            print(f"[{idx}] Processed: {base} → Output {idx}.png / {idx}.json, rotation {deg:.2f}°")
            idx += 1

    print(f"Finished processing {idx} images. Rotation angles saved to: {csv_path}")
