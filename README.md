
# Inference Guide: YOLOv8 Segmentation Overlay

This guide explains how to run the provided inference script and teeth whitening script locally to perform segmentation overlay and bounding‐box drawing on teeth images using a trained YOLOv8 model.

---

## 📋 Prerequisites

1. **Python**  
   - Version: **latest (3.8+)**


2. **Model Weights**  
   - Trained YOLOv8 `.pt` file (e.g. `best.pt`)

---

## 🔧 Installation

1. **Clone your repository**  
   ```bash
   git clone https://github.com/AnkitSharma1405/teeth_whitening.git
   ```

2. **Create & activate a virtual environment** (optional but recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate.bat   # Windows
   ```

3. **Install dependencies**  
   Make sure your `requirements.txt` contains:
   ```text
   ultralytics==8.1.0
   opencv-python==4.8.0.76
   numpy==1.23.5
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Script

**Run Inference**

```bash
python inference.py
```
 **Run teeth whitening**  
   ```bash
   python whitening_effect.py
   ```
## Output Effect:

**Input Image**
![test image](https://github.com/AnkitSharma1405/teeth_whitening/blob/100d5ffec14d5ab239bcb348f10ce529f65a1b6f/test_images/test_image.jpg)

**Inference Output**
![Inference](https://github.com/AnkitSharma1405/teeth_whitening/blob/100d5ffec14d5ab239bcb348f10ce529f65a1b6f/output_images/segmented_output.png)

**Whitening Effect**
![Teeth Whitening](https://github.com/AnkitSharma1405/teeth_whitening/blob/100d5ffec14d5ab239bcb348f10ce529f65a1b6f/output_images/teeth_whitening_02.png)

- **Ensure**:
  - `models/best.pt` exists or update `model = YOLO("…")` path.
  - `input/sample.jpg or input/sample.mp4` exists or point `input_path` to your image or video.
  - `output/segmented_output.jpg or output/sample_processed` will be created after program execution.

---

