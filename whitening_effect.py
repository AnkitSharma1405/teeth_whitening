from ultralytics import YOLO
import cv2
import numpy as np
import os

# ============================
# INFERENCE WITH TEETH-WHITENING OVERLAY
# ============================
# Load trained weights (best or last checkpoint)
model = YOLO(r"C:\Users\91637\Downloads\teeth_data_AL\models\best.pt")  # or 'last.pt'

# Function to overlay mask in light white for whitening effect
def overlay_whiten(image, masks, alpha=0.3):
    """
    Apply a light white overlay on masked regions to simulate whitening.
    image : np.ndarray (H, W, 3)
    masks : np.ndarray (N, Hm, Wm) boolean or uint8 mask array
    alpha : float transparency (0-1)
    """
    overlaid = image.copy()
    h_img, w_img = image.shape[:2]
    white = np.array([255, 255, 255], dtype=np.uint8)
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.uint8)
        # Resize mask to image size
        if mask.shape != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask.astype(bool)
        # Blend white color
        overlaid[mask_bool] = (
            overlaid[mask_bool] * (1 - alpha) + white * alpha
        ).astype(np.uint8)
    return overlaid

# Paths
input_path = r"C:\Users\91637\Downloads\teeth_data_AL\test_images\test_image.jpg"
output_dir = r"C:\Users\91637\Downloads\teeth_data_AL\output_images"
os.makedirs(output_dir, exist_ok=True)

# Run inference on CPU
device = 'cpu'
results = model.predict(
    source=input_path,
    conf=0.25,
    device=device,
    save=False
)

# Process first result
res = results[0]
# Read original image in BGR then convert to RGB
img_bgr = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# Extract masks
masks = res.masks.data.cpu().numpy()  # shape (N, Hm, Wm)
# Apply whitening overlay
overlaid_img = overlay_whiten(img_rgb, masks, alpha=0.4)

# Convert back to BGR and save
save_img = cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR)
output_path = os.path.join(output_dir, 'teeth_whitening_02.png')
cv2.imwrite(output_path, save_img)
print(f"Saved whitening effect image to {output_path}")

