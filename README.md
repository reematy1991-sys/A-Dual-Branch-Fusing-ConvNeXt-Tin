# ConvNeXt-HF: Hybrid Deep Learning for Skin Cancer Classification

A reliable, reproducible hybrid deep learning pipeline for classifying the seven diagnostic categories of the **HAM10000** dermoscopic dataset. This work fuses **ConvNeXt-Tiny** deep features with **handcrafted LBP + color-histogram** descriptors and enforces an **augmentation-leakage-free** evaluation protocol.

---

##  Overview

Skin cancer is among the most commonly diagnosed malignancies worldwide, and prognosis depends strongly on early detection. Dermoscopy is the clinical standard, but its accuracy is highly examiner-dependent and inter-observer agreement is imperfect. While deep CNNs have advanced computer-aided diagnosis (CAD), the literature still suffers from:

- **Augmentation leakage** — augmentation applied before train/test splitting, inflating reported accuracies.
- **Inconsistent and non-reproducible results** across studies.
- **Ignored class imbalance** in HAM10000.
- **Missing ablation studies** and no interpretable, offline deployment prototype.

**ConvNeXt-HF** addresses these gaps with a leakage-free protocol, a hybrid feature-fusion model, a four-stage ablation, and an offline GUI with Grad-CAM explainability.

---

##  Model Architecture

```text
                        ┌─────────────────────────┐
                        │   Dermoscopic Image     │
                        │      (224×224×3)        │
                        └────────────┬────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
   ┌───────────────────────────┐         ┌──────────────────────────────┐
   │   ConvNeXt-Tiny Backbone  │         │  Handcrafted Descriptors     │
   │  (pretrained, fine-tuned) │         │  • LBP                       │
   │  • Large-kernel depthwise │         │  • Color Histograms (HSV/RGB)│
   │  • Inverted bottlenecks   │         └──────────────┬───────────────┘
   │  • Layer normalization    │                        │
   └─────────────┬─────────────┘                        ▼
                 ▼                       ┌──────────────────────────────┐
   ┌───────────────────────────┐         │  Handcrafted Feature Vector  │
   │   Deep Feature Vector     │         │        (e.g., 256-D)         │
   │      (e.g., 768-D)        │         └──────────────┬───────────────┘
   └─────────────┬─────────────┘                        │
                 └──────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  Feature Fusion (Concat)      │
                    │   [ deep ‖ LBP ‖ color ]      │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Classification Head         │
                    │  FC → BN → ReLU → Dropout     │
                    │        → FC (7 classes)       │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 7-class Prediction            │
                    │ + Confidence Score (Softmax)  │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Grad-CAM Visualization      │
                    └───────────────────────────────┘
```




##  GUI Preview

The offline GUI provides a complete, clinician-friendly interface for real-time skin lesion analysis — **no internet connection required**.

### Features

-  **Drag-and-drop image upload** — load a dermoscopic image with a single gesture
-  **Real-time 7-class prediction** — instant inference across all HAM10000 categories (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)
-  **Grad-CAM overlay** — heatmap highlighting the diagnostically relevant regions that drove the model's decision
-  **Confidence score bar chart** — per-class softmax probabilities for transparent, interpretable outputs

### Screenshots


<img width="619" height="407" alt="image" src="https://github.com/user-attachments/assets/6c398689-9d05-425d-9f08-bcb468c63c0d" />


>  Replace the image paths above with your actual screenshots placed under `docs/images/`.

### Quick Launch

```bash
python gui/app.py
```

### Typical Workflow

```text
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Drop Image   │ ──► │  2. Run Model    │ ──► │  3. View Results │
│  (JPG / PNG)     │     │  (ConvNeXt-HF)   │     │  • Prediction    │
└──────────────────┘     └──────────────────┘     │  • Grad-CAM      │
                                                  │  • Confidence    │
                                                  └──────────────────┘
```
