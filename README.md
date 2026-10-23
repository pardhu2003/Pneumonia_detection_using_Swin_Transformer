::: {align="center"}
# 🫁 Pneumonia Detection from Chest X-Rays

### Deep Learning • Swin Transformer • Explainable AI

**An image-classification project for distinguishing NORMAL and
PNEUMONIA chest X-rays.**

```{=html}
<p>
```
`<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">`{=html}
`<img src="https://img.shields.io/badge/Model-Swin%20Transformer-6C5CE7?style=for-the-badge" alt="Swin Transformer">`{=html}
`<img src="https://img.shields.io/badge/Explainability-Grad--CAM%20%7C%20LIME%20%7C%20SHAP-00897B?style=for-the-badge" alt="Explainability">`{=html}
```{=html}
</p>
```
[📓 View Notebook](./pneumonia-detection-balanced.ipynb) • [📂
Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
:::

------------------------------------------------------------------------

## ✨ Project at a Glance

This project explores automated classification of chest X-ray images
using a **Swin Transformer Tiny** model. It combines image
preprocessing, training-set balancing, model evaluation, and
explainability methods to help inspect how the model produces
predictions.

> ⚠️ **Research and educational use only.** This model is not a medical
> device and must not be used to diagnose patients or make clinical
> decisions.

## 📊 Reported Test Results

The notebook's saved evaluation output reports results on a held-out
test set of **624 images**.

  Metric        NORMAL   PNEUMONIA
  ----------- -------- -----------
  Precision       0.85        0.85
  Recall          0.72        0.93
  F1-score        0.78        0.88
  Support          234         390

  Overall metric        Reported result
  ------------------- -----------------
  **Accuracy**               **84.94%**
  Log loss                       0.3423
  Macro F1-score                   0.83
  Weighted F1-score                0.85

**How to interpret this:** pneumonia recall is higher than NORMAL recall
in these results. Some NORMAL images may therefore be incorrectly
classified as PNEUMONIA. These results apply to this dataset split only
and do not establish performance in a clinical setting.

## 🧠 Project Pipeline

``` mermaid
flowchart TD
    A["Chest X-ray Dataset"] --> B["Dataset Inspection & Image Statistics"]
    B --> C["Conditional Image Enhancement"]
    C --> D["Resize to 224 × 224"]
    D --> E["Train / Validation Preparation"]
    E --> F["Augmentation-Based Training Balancing"]
    F --> G["Swin Transformer Tiny"]
    G --> H["Training & Validation"]
    H --> I["Held-Out Test Evaluation"]
    I --> J["Grad-CAM • LIME • SHAP"]
```

## 🚀 Key Features

-   🩻 **Binary classification:** NORMAL vs. PNEUMONIA.
-   🧹 **Image preprocessing:** quality checks, conditional denoising,
    CLAHE contrast enhancement, and resizing.
-   ⚖️ **Training-set balancing:** augmentation-based oversampling.
-   🧠 **Transformer model:** ImageNet-pretrained Swin Transformer Tiny
    with a trainable classification head.
-   📈 **Evaluation:** accuracy, precision, recall, F1-score, and log
    loss.
-   🔎 **Explainable AI:** Grad-CAM, LIME, and SHAP visualizations for
    exploring model behavior.
-   🎯 **Reproducibility:** fixed random seed and documented training
    settings.

## 🗂️ Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset.

-   **Source:** [Kaggle --- Chest X-Ray Images
    (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
-   **Classes:** `NORMAL`, `PNEUMONIA`
-   **Images scanned by the notebook:** 5,856
-   **Prepared split reported in the notebook:** 4,708 training, 524
    validation, and 624 test images

Expected directory structure:

``` text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> The dataset is not included in this repository. Download it from
> Kaggle and make it available in your notebook environment. Image
> counts can vary depending on the dataset copy and preparation steps.

## 🛠️ Technology Stack

  Area                   Tools
  ---------------------- -----------------------
  Language               Python
  Deep learning          PyTorch, Torchvision
  Image processing       OpenCV, Pillow
  Data handling          NumPy, Pandas
  Visualization          Matplotlib, Seaborn
  Evaluation             scikit-learn
  Explainability         Grad-CAM, LIME, SHAP
  Notebook environment   Google Colab / Kaggle

## ⚙️ Getting Started

### Option 1 --- Google Colab

1.  Clone or download this repository.
2.  Open `pneumonia-detection-balanced.ipynb` in Google Colab.
3.  Download the dataset from
    [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
4.  Make the dataset folders accessible to the notebook and update the
    path if needed.
5.  Select a GPU runtime if available, then run the notebook cells in
    order.

### Option 2 --- Local environment

Install the main dependencies:

``` bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn pillow tqdm grad-cam lime shap
```

Use a PyTorch/Torchvision installation compatible with your Python and
CUDA versions. Then open the notebook in Jupyter and ensure the dataset
path matches your local setup.

## 🔬 Model Configuration

  Parameter             Value
  --------------------- ---------------------------
  Architecture          Swin Transformer Tiny
  Initial weights       ImageNet-pretrained
  Input size            224 × 224
  Batch size            32
  Learning rate         0.0001
  Epochs                5
  Optimizer             Adam
  Trainable component   Final classification head
  Random seed           42
  Output classes        2

## 🔍 Explainability

  -----------------------------------------------------------------------
  Method                              Purpose
  ----------------------------------- -----------------------------------
  **Grad-CAM**                        Highlights image regions associated
                                      with a model output.

  **LIME**                            Estimates how local image segments
                                      influence a prediction.

  **SHAP**                            Estimates feature contributions to
                                      model output.
  -----------------------------------------------------------------------

These methods are useful for exploratory analysis, but their
visualizations do not prove that a prediction is correct or that the
model uses medically meaningful evidence.

## 📁 Repository Structure

``` text
pneumonia-detection-swin-transformer/
├── README.md
└── pneumonia-detection-balanced.ipynb
```

## ⚠️ Limitations & Responsible Use

-   Results are based on one held-out dataset split; external validation
    is not reported.
-   Performance may change across hospitals, scanners, patient
    populations, and image-processing pipelines.
-   Recall differs between the two classes, so accuracy alone is
    insufficient to assess performance.
-   Explainability maps can be unstable or misleading.
-   **Do not use this model for diagnosis, treatment, triage, or other
    clinical decisions.**

## 💡 Future Improvements

-   Compare the model against a CNN baseline and fine-tune selected
    transformer layers.
-   Add a confusion matrix, ROC-AUC, PR-AUC, sensitivity, specificity,
    and confidence intervals.
-   Evaluate on an independent dataset and investigate possible
    patient-level data leakage.
-   Add a reproducible inference script and pinned dependency versions.
-   Review explanations with qualified domain experts.

## 📚 Dataset Credit

Please review and follow the dataset's terms of use.

-   Kermany et al., *Identifying Medical Diagnoses and Treatable
    Diseases by Image-Based Deep Learning*. **Cell (2018)**.
-   [Chest X-Ray Images (Pneumonia) ---
    Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

------------------------------------------------------------------------

::: {align="center"}
**Built for learning, experimentation, and responsible AI research.**

⭐ If you find this project useful, consider starring the repository.
:::
