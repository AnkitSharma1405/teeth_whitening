
# YOLOv8-Based Tooth Detection Model

## 📄 Detailed Documentation

---

### 🔍 Model Selection

- **Familiarity and Experience with YOLOv8**:  
  My prior experience with YOLOv8 enabled efficient data preprocessing and streamlined the overall workflow.

- **Advantages Over Other Models**:  
  YOLOv8 offers faster inference times and handles complex scenarios better—such as detecting small objects.

---

### 🧹 Data Preprocessing for YOLOv8

YOLOv8 expects:

- A specific directory structure
- Images paired with `.txt` annotation files (YOLO format)
- Annotations containing normalized bounding box coordinates

However, the provided dataset:

- Had a different folder structure
- Contained annotations in **JSON** format
- Used **polygon coordinates**

To resolve this, I used **[Roboflow](https://roboflow.com/)** for preprocessing and conversion.

#### ✅ Step 1: Annotation Conversion

- Uploaded the dataset to Roboflow
- Used annotation conversion tools to transform **polygon-based JSON** annotations into **bounding boxes**
- Exported data in YOLO format (`.txt` files with class label + normalized bounding box coordinates)

#### ✅ Step 2: Image Preprocessing

- Resized all images to **640×640** pixels using Roboflow

#### ✅ Step 3: Data Augmentation

Applied augmentations using Roboflow to increase dataset size by 3×:
- Horizontal and Vertical Flips
- Blur Effects

---

### 🏷️ Handling Four Classes

To simplify the task:
- Wrote a Python script (`modify_annotations.py`) to:
  - Remove labels for all classes **except** `"Tooth"`
  - Reassign `"Tooth"` label from `3` to `0`

This helped focus the dataset on the required class and maintain model compatibility.

---

### 🧠 Model Training

The model was trained on ~6,000 images with the following optimizations:

- **Parallel Data Loading**:  
  `num_workers=4` for efficient batch handling

- **Batch Size**:  
  `batch_size=8`—a trade-off between memory usage and training stability

- **Early Stopping**:  
  Used `patience=20` to prevent overfitting and save compute time

- **Checkpointing**:  
  Saved model weights every **15 epochs** to enable recovery in case of interruption

---

### 🚀 Potential Future Improvements

- **Attention Mechanisms**:  
  Integrate attention modules to better focus on the teeth region vs. background

- **Detection Block Modification**:  
  Modify YOLOv8 architecture to **ignore large object detection**, making the model more lightweight

- **Advanced Augmentations**:  
  Apply more transformations like **brightness adjustment** and **rotation** to improve generalization

---



