Here is a complete, polished **README.md** specifically tailored for this **Histopathological Lung Cancer Image Classification** project, following the same structure and formatting as your previous PyTorch README.

---

# Deep Learning Pipeline for Histopathological Lung Cancer Classification

This repository contains a PyTorch implementation of a deep learning pipeline using transfer learning (ResNet-50) to classify histopathological tissue images for automated lung cancer diagnosis.

---

## 📌 Overview & Business Relevance

Histopathological imaging is the gold standard for cancer diagnosis, but manual analysis is time-intensive and subject to inter-observer variability. This project delivers an automated computer vision pipeline capable of classifying high-resolution tissue slides into three distinct categories:

* **`Lung_acaR`**: Lung Adenocarcinoma
* **`Lung_sccR`**: Lung Squamous Cell Carcinoma
* **`Lung_nR`**: Benign (Normal) Lung Tissue

By pre-screening and auto-labeling diagnostic slides, this system accelerates pathology workflows and provides consistent diagnostic support in clinical settings.

---

## 🔬 Key Pipeline Highlights

* **Dataset Strategy:** Stratified splitting across training (5,014), validation (558), and testing (984) sets to maintain class proportions.
* **Data Augmentation:** Implemented `RandomResizedCrop`, horizontal/vertical flips, and `ColorJitter` to prevent overfitting.
* **Transfer Learning:** Fine-tuned an ImageNet-pretrained **ResNet-50** architecture with custom linear output layers.
* **Optimization:** Trained using Cross-Entropy Loss and Adam optimizer with model checkpointing based on validation accuracy.

---

## 📊 Performance & Metrics

The model achieved high predictive precision and discriminative power across all evaluation metrics on an independent test set of **984 images**:

| Metric | Score |
| --- | --- |
| **Overall Test Accuracy** | **95.73%** |
| **Weighted Precision** | **95.77%** |
| **Weighted Recall** | **95.73%** |
| **Weighted F1-Score** | **95.75%** |
| **Weighted ROC-AUC (OvR)** | **99.66%** |

### Per-Class Performance

```text
              precision    recall  f1-score   support

   Lung_acaR       0.92      0.92      0.92       264
     Lung_nR       1.00      0.99      0.99       363
   Lung_sccR       0.94      0.95      0.95       357

    accuracy                           0.96       984
   macro avg       0.95      0.95      0.95       984
weighted avg       0.96      0.96      0.96       984

```

---

## 🚀 Quick Start

### 1. Run in Google Colab

Run and train the notebook directly in Google Colab:

### 2. Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Rominaarab/teaching.git
cd teaching

```


2. **Install required dependencies:**
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow

```


3. **Data Preparation:**
Ensure your compressed dataset (`Lung R.zip`) is present in the working directory or update the `zip_path` variable inside the notebook.

---

## 🛠️ Tech Stack & Dependencies

* **Deep Learning Framework:** PyTorch, `torchvision`
* **Data Processing & Analysis:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Image Handling:** PIL (Pillow)

---

## 📚 Code Highlights

### Pre-trained ResNet-50 Adaptation

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace final classification head for 3 lung tissue classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

```

---

## 📖 Reference & Dataset Source

* **Dataset Paper:** *LC25000: Lung and Colon Cancer Histopathological Image Dataset* — [arXiv:1912.12142](https://arxiv.org/abs/1912.12142v1)

---

## 👤 Author

**Romina Arab**

* GitHub: [@Rominaarab](https://www.google.com/search?q=https://github.com/Rominaarab)
