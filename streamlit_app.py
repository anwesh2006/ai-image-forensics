import os
import cv2
import gdown
import numpy as np
import streamlit as st
import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Synthetic Image Forensics",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Synthetic / AI-Generated Image Forensics")
st.markdown(
    "Explainable AI detection powered by **ConvNeXt-Tiny**, **Grad-CAM Saliency**, "
    "and **2D Fast Fourier Transform (FFT)** frequency analysis."
)

# ---------------------------------------------------------
# Model & Weights Configuration
# ---------------------------------------------------------
DEVICE = "cpu"
MODEL_PATH = "best_detector_convnext.pth"

# Paste your Google Drive File ID here:
DRIVE_FILE_ID = "10xEA2w3dbTFeULpCSAU8OHEU9LXgOnCC"


def download_weights_if_missing():
    """Downloads model weights directly from Google Drive if not present locally."""
    if not os.path.exists(MODEL_PATH):
        if DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID":
            st.error("Please replace `YOUR_GOOGLE_DRIVE_FILE_ID` in `streamlit_app.py` with your actual file ID.")
            st.stop()
        with st.spinner("Downloading trained model weights (~106 MB)..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)


download_weights_if_missing()


@st.cache_resource
def load_detector():
    """Loads and caches the ConvNeXt-Tiny model on CPU."""
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_detector()
target_layers = [model.stages[-1].blocks[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
targets = [BinaryClassifierOutputTarget(None)]

# ---------------------------------------------------------
# Forensics & Image Processing Helper Functions
# ---------------------------------------------------------
def center_crop(img):
    """Crops the center square to preserve aspect ratio without stretching."""
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return img[start_y:start_y + min_dim, start_x:start_x + min_dim]


def compute_fft_spectrum(img_rgb):
    """Generates the 2D FFT magnitude spectrum to expose high-frequency lattice patterns."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(np.uint8(norm), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# UI Sidebar & Controls
# ---------------------------------------------------------
st.sidebar.header("Inference Controls")
match_cifake_scale = st.sidebar.checkbox(
    "Align with CIFAKE Resolution (32x32 -> 224x224)",
    value=True,
    help="Matches the exact CIFAR-10 low-resolution interpolation format the model was trained on."
)

temperature = st.sidebar.slider(
    "Temperature Calibration (T)",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help="Higher values soften extreme out-of-domain confidence spikes."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset Context:**\n"
    "Trained on CIFAKE (60,000 real CIFAR-10 images vs. 60,000 Stable Diffusion v1.4 generations). "
    "Unseen categories (e.g. food, text, human faces) may display lower certainty."
)

# ---------------------------------------------------------
# Image Upload & Processing Pipeline
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image to inspect (JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # 1. Aspect-ratio preserving square crop
    cropped = center_crop(raw_rgb)

    # 2. Strict CIFAKE training distribution matching (no artificial JPEG block noise)
    if match_cifake_scale:
        cifar_scale = cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(cifar_scale, (224, 224), interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 3. Normalization for ConvNeXt
    float_img = np.float32(resized) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    tensor = (float_img - mean) / std
    tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 4. Model Inference
    with torch.no_grad():
        raw_logit = model(tensor).squeeze()
        calibrated_logit = raw_logit / temperature
        # ImageFolder alphabetical sorting: 0 = FAKE, 1 = REAL
        prob_real = torch.sigmoid(calibrated_logit).item()
        prob_fake = 1.0 - prob_real

    # 5. Explainability Map Generation
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(float_img, grayscale_cam, use_rgb=True)
    fft_image = compute_fft_spectrum(resized)

    # 6. Calibrated Thresholding (mitigates false positive synthetic bias)
    if prob_fake >= 0.85:
        verdict = "AI-GENERATED (SYNTHETIC)"
        status_color = "red"
        confidence = prob_fake
    elif prob_fake <= 0.45:
        verdict = "AUTHENTIC (REAL)"
        status_color = "green"
        confidence = prob_real
    else:
        verdict = "UNCERTAIN / OUT-OF-DOMAIN"
        status_color = "orange"
        confidence = max(prob_fake, prob_real)

    # ---------------------------------------------------------
    # Render Diagnostics & Visualizations
    # ---------------------------------------------------------
    st.markdown(f"### Verdict: :{status_color}[**{verdict}**]")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Confidence", f"{confidence * 100:.2f}%")
    m_col2.metric("Synthetic Probability", f"{prob_fake:.4f}")
    m_col3.metric("Authentic Probability", f"{prob_real:.4f}")

    st.markdown("---")
    st.subheader("Explainable AI (XAI) Forensic Evidence")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(resized, caption="1. Normalized Input (Center Cropped)", use_container_width=True)
    with col2:
        st.image(cam_image, caption="2. Grad-CAM Saliency Map", use_container_width=True)
    with col3:
        st.image(fft_image, caption="3. 2D FFT Frequency Spectrum", use_container_width=True)

    with st.expander("Interpret Forensics Evidence"):
        st.markdown(
            "- **Grad-CAM Activation:** Highlights the specific spatial patches that drove the classification decision. "
            "Generative images often trigger activations along smoothed boundaries, specular gloss, or structural anomalies.\n"
            "- **2D FFT Spectrum:** Displays high-to-low spatial frequency distribution. Synthetic generators typically exhibit "
            "abnormal cross-axis spikes or regular lattice grids from transposed convolution upsampling."
        )
