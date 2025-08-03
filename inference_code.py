from ultralytics import YOLO
import cv2
import numpy as np
import os

# ============================
# INFERENCE WITH SEGMENT OVERLAY (CPU)
# ============================
# Load trained weights (best or last checkpoint)
model = YOLO(r"C:\Users\91637\Downloads\teeth_data_AL\models\best.pt")  # or 'last.pt

# Function to overlay mask on image
def overlay_masks(image, masks, alpha=0.5, color_map=None):
    """
    image : np.ndarray (H, W, 3)
    masks : np.ndarray (N, Hm, Wm) boolean mask array (model output size)
    alpha : float transparency factor for overlay
    """
    overlaid = image.copy()
    h_img, w_img = image.shape[:2]
    n_masks = masks.shape[0]
    # Generate random colors
    rng = np.random.default_rng(42)
    color_map = [tuple(int(c) for c in rng.integers(0, 255, size=3)) for _ in range(n_masks)]

    for i in range(n_masks):
        mask = masks[i].astype(np.uint8)
        # Resize mask to image size if needed
        if mask.shape != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask.astype(bool)
        color = color_map[i]
        # Overlay color on masked regions
        overlaid[mask_bool] = (overlaid[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlaid

# Paths
input_path = r"C:\Users\91637\Downloads\teeth_data_AL\test_images\test_image.jpg"
output_dir = r"C:\Users\91637\Downloads\teeth_data_AL\output_images"
os.makedirs(output_dir, exist_ok=True)

# Run inference on CPU
results = model.predict(
    source=input_path,
    conf=0.25,
    device='cpu',
    save=False
)

# Process the first result
res = results[0]
# Read original image in RGB
img_bgr = cv2.imread(input_path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Extract masks and overlay
masks = res.masks.data.cpu().numpy()  # shape (N, Hm, Wm)
overlaid_img = overlay_masks(img, masks, alpha=0.5)

# Draw bounding boxes and labels
for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(overlaid_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(overlaid_img, f"Class {int(cls)}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save output (convert back to BGR)
save_img = cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR)
output_path = os.path.join(output_dir, 'segmented_output_01.png')
cv2.imwrite(output_path, save_img)
print(f"Saved overlaid segmentation to {output_path}")

# Optional display
# cv2.imshow('Segmentation Overlay', save_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





