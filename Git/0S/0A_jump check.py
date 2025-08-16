import os
import cv2

def detect_jump_images(image_dir, threshold=10):
    # Gather all image file paths with common extensions and sort them
    image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.endswith((".tif", ".png", ".jpg"))
    ])
    
    prev_size = None
    for path in image_paths:
        # Load the image in grayscale mode
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[⚠️] Failed to read image: {path}")
            continue
        # Get current image dimensions (height, width)
        h, w = img.shape
        if prev_size:
            ph, pw = prev_size
            # Detect significant size change compared to the previous image
            if abs(h - ph) > threshold or abs(w - pw) > threshold:
                print(
                    f"Jump detected: {os.path.basename(path)} "
                    f"({h}x{w}) vs previous ({ph}x{pw})"
                )
        # Update previous size for next comparison
        prev_size = (h, w)

# Example usage
if __name__ == "__main__":
    detect_jump_images(
        image_dir=r"D:\Study\Postgraduate\S2\Project\R\Important materials for  Dissertation\Raw data part\ALLTIF\13_14_15_16\16",
        threshold=5  # Adjust the threshold for detecting size jumps
    )
