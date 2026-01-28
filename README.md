# A Tumor-Aware Hybrid ConvNeXt–Swin U-Net for Brain MRI Segmentation and Classification

## 📌 Overview
This repository presents a **tumor-aware deep learning framework for brain MRI analysis**, integrating a **hybrid ConvNeXt–Swin U-Net architecture** for accurate tumor segmentation and subsequent tumor type classification. The proposed approach combines **CNN-based local feature extraction** with **transformer-based global context modeling**, followed by **ROI-guided classification** to reduce background bias and improve reliability.

The framework is designed as a **two-stage pipeline**:
1. Pixel-wise brain tumor segmentation  
2. Tumor-aware classification using only segmented tumor regions  

Experiments are conducted on the **BRISC 2025 dataset**, demonstrating improved segmentation performance in terms of **Dice coefficient and IoU**, along with robust multi-class tumor classification.

---

## 🧠 Key Contributions
- A **hybrid ConvNeXt–Swin U-Net** architecture that jointly models local texture details and global contextual information
- **Multi-scale feature fusion** for improved tumor boundary delineation
- **Tumor-aware ROI-based classification** to reduce background influence
- Fair and systematic comparison of CNN-based, transformer-based, and hybrid segmentation models
- Clinically meaningful evaluation using overlap-based metrics

---

## 📊 Dataset
- **Dataset:** BRISC 2025  
- **Modality:** 2D T1-weighted brain MRI slices  
- **Input Size:** 256 × 256  
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor  
- **Split:**  
  - Training: 5,000 images  
  - Testing: 1,000 images  
- **Annotations:** Expert-provided ground-truth segmentation masks  
- **Class Distribution:** Balanced  

---

## 🔄 Overall Pipeline

1. **Preprocessing**
   - Image resizing to 256 × 256
   - RGB conversion
   - Intensity normalization using ImageNet mean and standard deviation
   - Binary mask preparation and visual sanity checks

2. **Tumor Segmentation**
   - Baseline U-Net
   - ConvNeXt-UNet
   - Swin-UNet
   - **Proposed Hybrid ConvNeXt–Swin U-Net**

3. **Tumor-Aware Feature Extraction**
   - ROI extraction using predicted segmentation masks
   - Feature computation limited to tumor regions

4. **Tumor Type Classification**
   - Multi-class classification using tumor-specific deep features

---

## 🧠 Models Used

### 🔹 U-Net (Baseline)
A standard CNN-based U-Net architecture used as a baseline for segmentation comparison.

### 🔹 ConvNeXt-UNet
Employs a ConvNeXt-Tiny encoder pretrained on ImageNet, offering improved local feature representation and better tumor boundary localization.

### 🔹 Swin-UNet
Integrates a Swin Transformer encoder with shifted-window attention to capture long-range contextual dependencies.

### 🔹 Proposed Hybrid ConvNeXt–Swin U-Net
Combines ConvNeXt and Swin Transformer encoders with **multi-scale feature fusion**, enabling robust segmentation by leveraging both local and global feature modeling.

---

## 📐 Evaluation Metrics

### Segmentation Metrics
- **Dice Coefficient (Primary Metric):** Measures overlap between predicted and ground-truth masks and is well-suited for medical image segmentation with class imbalance.
- **Intersection over Union (IoU):** Quantifies segmentation quality by comparing intersection and union regions.
- **Pixel-wise Accuracy:** Reported as a supporting metric.

### Classification Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 📈 Key Observations
- ConvNeXt-UNet outperforms the baseline U-Net, highlighting the effectiveness of modern CNN designs.
- Swin-UNet captures global context but shows slightly reduced performance on fine tumor boundaries in 2D MRI slices.
- The **Hybrid ConvNeXt–Swin U-Net achieves the best Dice and IoU scores**, demonstrating the advantage of combining CNN and transformer features.
- ROI-based tumor-aware classification reduces background bias and improves class discrimination.

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- NumPy, OpenCV  
- Matplotlib / Seaborn  

---

## 📁 Project Structure
