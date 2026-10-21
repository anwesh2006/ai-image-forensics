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
DRIVE_FILE_ID = "10xEA2w3dbTFeULpCSAU8OHEU9LXgOnCC"


def download_weights_if_missing():
    """Downloads model weights directly from Google Drive if not present locally."""
    if not os.path.exists(MODEL_PATH):
        if DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID":
            st.error("Please replace `YOUR_GOOGLE_DRIVE_FILE_ID` with your actual file ID.")
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
    """Crops center square to preserve aspect ratio without non-uniform stretching."""
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
# Sidebar Diagnostic & Tuning Controls
# ---------------------------------------------------------
st.sidebar.header("Forensics Calibration")

invert_classes = st.sidebar.checkbox(
    "Invert Prediction Polarity",
    value=False,
    help="Enable if your training setup treated 1 as FAKE instead of REAL."
)

resolution_mode = st.sidebar.selectbox(
    "Input Resolution Processing",
    options=["Native 224x224 (Preserves AI Artifacts)", "CIFAR Scale 32x32 (Dampens Camera Noise)"],
    index=0,
    help="Switch to Native 224x224 if synthetic images are slipping through as real."
)

detection_profile = st.sidebar.radio(
    "Detection Sensitivity Profile",
    options=["Balanced (Threshold 0.50)", "High AI Sensitivity (Threshold 0.35)", "High Authentic Sensitivity (Threshold 0.65)"],
    index=0,
    help="Lower threshold catches subtle AI generations; higher threshold demands stronger proof before flagging AI."
)

temperature = st.sidebar.slider(
    "Logit Calibration Temperature (T)",
    min_value=1.0,
    max_value=2.5,
    value=1.0,
    step=0.1,
    help="Keep at 1.0 for raw training confidence. Increase only if predictions are overly aggressive."
)

# Parse selected threshold
if "High AI Sensitivity" in detection_profile:
    active_threshold = 0.35
elif "High Authentic Sensitivity" in detection_profile:
    active_threshold = 0.65
else:
    active_threshold = 0.50

# ---------------------------------------------------------
# Upload & Inference Loop
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image to inspect (JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # 1. Center crop
    cropped = center_crop(raw_rgb)

    # 2. Resizing strategy
    if resolution_mode == "CIFAR Scale 32x32 (Dampens Camera Noise)":
        small = cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(small, (224, 224), interpolation=cv2.INTER_CUBIC)
    else:
        # Native direct resize preserves microscopic diffusion generator artifacts
        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 3. Standard ImageNet Normalization
    float_img = np.float32(resized) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    tensor = (float_img - mean) / std
    tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        raw_logit = model(tensor).squeeze().item()
        calibrated_logit = raw_logit / temperature
        p = torch.sigmoid(torch.tensor(calibrated_logit)).item()

        if invert_classes:
            prob_fake = p
            prob_real = 1.0 - p
        else:
            prob_real = p
            prob_fake = 1.0 - p

    # 5. XAI Visualizations
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(float_img, grayscale_cam, use_rgb=True)
    fft_image = compute_fft_spectrum(resized)

    # 6. Verdict Calculation
    if prob_fake >= active_threshold:
        verdict = "AI-GENERATED (SYNTHETIC)"
        status_color = "red"
        confidence = prob_fake
    else:
        verdict = "AUTHENTIC (REAL)"
        status_color = "green"
        confidence = prob_real

    # Render Results
    st.markdown(f"### Verdict: :{status_color}[**{verdict}**]")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{confidence * 100:.2f}%")
    m2.metric("Synthetic Probability", f"{prob_fake:.4f}")
    m3.metric("Authentic Probability", f"{prob_real:.4f}")
    m4.metric("Raw Logit (z)", f"{raw_logit:+.2f}")

    st.markdown("---")
    st.subheader("Explainable AI (XAI) Forensic Evidence")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(resized, caption="1. Analyzed Crop (224x224)", use_container_width=True)
    with c2:
        st.image(cam_image, caption="2. Grad-CAM Activation Heatmap", use_container_width=True)
    with c3:
        st.image(fft_image, caption="3. 2D FFT Frequency Spectrum", use_container_width=True)

    with st.expander("Diagnostics & Troubleshooting"):
        st.write(
            f"- **Raw Logit Interpretation:** Logit = `{raw_logit:+.4f}`. "
            f"If an obvious AI image gives a strong positive logit (e.g. > +2.0), toggle **'Invert Prediction Polarity'** in the sidebar."
        )
        st.write(
            "- If modern generators (Flux, Midjourney, SDXL) slip through as real, select **'High AI Sensitivity (Threshold 0.35)'** "
            "because newer diffusion models produce fewer low-frequency errors than SD v1.4."
        )
