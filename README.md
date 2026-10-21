# Explainable AI (XAI) Synthetic Image Forensics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://share.streamlit.io](https://ai-image-forensics1.streamlit.app/))
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and explainability framework designed to classify authentic vs. synthetic (AI-generated) imagery. Built on a modern **ConvNeXt-Tiny** backbone, the pipeline integrates spatial saliency mapping (**Grad-CAM**) and spectral decomposition (**2D Fast Fourier Transform**) to expose both semantic and high-frequency structural signatures.

The project features a decoupled cloud architecture deployed on **Streamlit Community Cloud**, complete with dynamic weight downloading, logit calibration, and a diagnostic evaluation of **shortcut learning and out-of-distribution (OOD) generalization failures**.

---

## Table of Contents
- [Architecture & Key Features](#architecture--key-features)
- [Dataset & Model Training](#dataset--model-training)
- [Explainability Pipeline](#explainability-pipeline)
  - [Grad-CAM Saliency Maps](#1-grad-cam-spatial-activations)
  - [2D FFT Spectral Analysis](#2-2d-fft-magnitude-spectrum)
- [Analytical Findings: The Shortcut Learning Dilemma](#analytical-findings-the-shortcut-learning-dilemma)
- [System Architecture & Deployment](#system-architecture--deployment)
- [Installation & Local Run](#installation--local-run)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Architecture & Key Features

* **Backbone Architecture:** Modern pure-convolutional **ConvNeXt-Tiny** initialized with ImageNet-1K pre-trained representations and fine-tuned for single-logit binary classification.
* **Dual Explainability Streams:**
  * Spatial domain: Class Activation Mapping via Gradient-weighted Class Activation Mapping (Grad-CAM) targeted at final-stage convolutional blocks (`stages[-1].blocks[-1]`).
  * Frequency domain: 2D Discrete Fast Fourier Transform ($\text{FFT}$) magnitude spectrum extraction to visualize lattice anomalies and checkerboard artifacts caused by transposed convolution and upsampling.
* **Decoupled Cloud Inference:** Resolves Git repository file-size constraints ($>100\text{ MB}$ weights) by fetching model checkpoints dynamically on container initialization from Google Drive via `gdown`.
* **Post-Processing Calibration:** Configurable temperature scaling ($T \in [1.0, 3.0]$) and sensitivity profiling to evaluate probability drift and calibration under domain shift.

---

## Dataset & Model Training

The classifier was trained on the **CIFAKE** benchmark dataset, composed of 120,000 $32 \times 32$ images split equally between authentic and synthetic classes across 10 object categories (*airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*).

* **Authentic Class:** Native CIFAR-10 photographic samples collected from real-world captures.
* **Synthetic Class:** Latent-diffusion generations generated with **Stable Diffusion v1.4**, prompt-conditioned on CIFAR-10 class labels and resampled.
* **Input Resolution:** Upscaled to $224 \times 224$ with bicubic interpolation and normalized using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### Training Configuration & Metrics

| Hyperparameter / Metric | Value |
|---|---|
| **Base Architecture** | `convnext_tiny` (28.6M parameters) |
| **Loss Function** | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-2}$) |
| **Learning Rate Schedule** | Cosine Annealing ($lr_0 = 10^{-4}$, $lr_{min} = 10^{-6}$) |
| **Test Set Accuracy** | **98.48%** |
| **Test Set AUC-ROC** | **0.9986** |

---

## Explainability Pipeline

Modern generative forensics requires verifiable reasoning rather than black-box score generation.

```
       [ Uploaded Image ]
               |
      +--------+--------+
      |                 |
      v                 v
[ Spatial Domain ]   [ Frequency Domain ]
      |                 |
ConvNeXt-Tiny       2D Discrete FFT
      |                 |
  Grad-CAM       Shifted Magnitude
  Heatmap             Spectrum
      \                 /
       v               v
   [ Interactive Diagnostic View ]
```

### 1. Grad-CAM Spatial Activations

Grad-CAM calculates the gradient of the unnormalized class score $y^c$ with respect to the feature activation map $A^k$ of the final convolutional stage:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

This visualizes whether the decision was driven by semantic structural features (e.g., eye asymmetry, unnatural specular reflections) or uninterpretable boundary noise.

### 2. 2D FFT Magnitude Spectrum

Generative architectures (such as GANs and Latent Diffusion models) introduce subtle periodicity in pixel space through deconvolutions and latent decoding. The 2D Discrete Fourier Transform converts luminance $I(x, y)$ to frequency domain $F(u, v)$:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \cdot e^{-j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$

$$\text{Spectrum}(u, v) = 20 \cdot \log_{10}\left( |F_{\text{shifted}}(u, v)| + \epsilon \right)$$

High-frequency cross-axis energy spikes, concentric rings, or structural grids on the inferno colormap indicate synthetic reconstruction fingerprints.

---

## Analytical Findings: The Shortcut Learning Dilemma

During deployment evaluations on arbitrary external images (such as modern smartphone photos, web images, and unseen Midjourney/Flux generations), the model revealed a fundamental challenge in deep synthetic forensics: **Shortcut Learning**.

### Empirical Observations
1. **False-Positive Bias on Raw High-Resolution Captures:**
   When given modern uncompressed camera photos, the model frequently output synthetic probabilities exceeding $95\%$.
2. **False-Negative Bias on Low-Pass Filtered Images:**
   Downsampling or smoothing images suppressed the false positives, but simultaneously erased subtle diffusion artifacts, causing synthetic samples to be labeled as authentic.

### Root-Cause Breakdown
* **Frequency Disparity:** Real CIFAR-10 images originate from 2009-era collection pipelines with distinct low-frequency profiles and legacy JPEG quantization tables.
* **Latent Decoder Signatures:** Stable Diffusion v1.4 maps $512 \times 512$ latents through an $8\times$ autoencoder before resizing to $32 \times 32$.
* **Representation Exploitation:** Deep convolutional networks inherently minimize loss along the path of least resistance. Rather than learning complex semantic rules, ConvNeXt discovered that **sensor noise profiles and interpolation kernels** separated the two classes with $99.8\%$ accuracy. When confronted with out-of-distribution modern sensor noise, the model interpreted that noise floor as synthetic.

This finding demonstrates that near-perfect accuracy on synthetic benchmarks does not imply real-world generalization without multi-generator diversity and aggressive anti-shortcut augmentations (such as randomized JPEG compression and Gaussian filtering).

---

## System Architecture & Deployment

The deployment pipeline is built to operate under strict free-tier resource limits:

```
[ Visitor Browser ]
        |
        v
[ Streamlit Community Cloud (CPU Runtime) ]
        |
        |--- (Check: best_detector_convnext.pth exists?)
        |         |-- Yes -> Load into RAM via @st.cache_resource
        |         +-- No  -> Stream from Google Drive (gdown)
        |
        +--> Center-Crop & Normalization
        +--> Forward Pass & Temperature Scaling: z_cal = z / T
        +--> Saliency & Spectral Processing (Grad-CAM & 2D FFT)
```

* **Storage Decoupling:** Bypasses GitHub's $25\text{ MB}$ browser limit and Git LFS bandwidth restrictions by hosting the $106\text{ MB}$ checkpoint externally on Google Drive.
* **Memory Management:** The network state dictionary is loaded once and cached via `@st.cache_resource`, ensuring subsequent web sessions incur zero re-download or memory duplication overhead.
* **Inference Runtime:** Fully optimized for CPU inference with an average execution latency of under $450\text{ ms}$ per sample.

---

## Installation & Local Run

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/synthetic-image-forensics.git
cd synthetic-image-forensics

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run streamlit_app.py
```

---

## Repository Structure

```text
├── streamlit_app.py       # Main application interface, pipeline orchestration & XAI
├── requirements.txt       # Production dependencies for container runtime
├── README.md              # Project documentation and forensic failure analysis
└── .gitignore             # Standard git exclusions (cache, local weights, venv)
```

---

## Requirements

```text
streamlit
torch
torchvision
timm
grad-cam
opencv-python-headless
numpy
gdown
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
