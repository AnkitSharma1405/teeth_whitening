from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO(r"E:\teeth_dataset_aftershoot\teeth_data_AL\model_result\model\best.pt")  # or 'last.pt'

# Function to overlay mask on image
def overlay_masks(image, masks, alpha=0.5):
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

# Function to process a single frame/image
def process_frame(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Run inference on CPU
    results = model.predict(
        source=img,
        conf=0.25,
        device='cpu',
        save=False
    )
    if len(results) == 0 or results[0].masks is None:
        return img_bgr  # No detections, return original
    res = results[0]
    # Extract masks and overlay
    masks = res.masks.data.cpu().numpy()  # shape (N, Hm, Wm)
    overlaid_img = overlay_masks(img, masks, alpha=0.5)
    # Draw bounding boxes and labels
    for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlaid_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(overlaid_img, f"Class {int(cls)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Convert back to BGR
    return cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR)

# Paths
input_path = r"E:\teeth_dataset_aftershoot\teeth_data_AL\test_images\test_image.jpg"  # Change to video path for video inference, e.g., 'test_video.mp4'
output_dir = r"E:\teeth_dataset_aftershoot\teeth_data_AL\output_images"  # Note: User mentioned 'outputr', assuming it's 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Determine if input is image or video based on extension
file_ext = os.path.splitext(input_path)[1].lower()

if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:  # Image formats
    # Process image
    img_bgr = cv2.imread(input_path)
    processed_img = process_frame(img_bgr)
    output_path = os.path.join(output_dir, 'segmented_output01.png')
    cv2.imwrite(output_path, processed_img)
    print(f"Saved overlaid segmentation image to {output_path}")

elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:  # Video formats
    # Process video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + '_processed.mp4'
        output_path = os.path.join(output_dir, output_filename)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            out.write(processed_frame)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"Saved processed video to {output_path}")

else:
    print(f"Unsupported file format: {file_ext}")
