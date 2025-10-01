import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter

st.title("🖼️ Image Noise Removal App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Dropdown for noise type
noise_type = st.selectbox(
    "Select Noise to Remove",
    ["Salt and pepper Noise", "Gaussian Noise", "White Noise", "Multiplicative Noise", "Quantization Noise"]
)

def remove_noise(img, noise_type):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if noise_type == "Salt and pepper Noise":
        # Median filter is effective
        denoised = median_filter(img_gray, size=3)

    elif noise_type == "Gaussian Noise":
        # Gaussian filter
        denoised = cv2.GaussianBlur(img_gray, (5, 5), 1)

    elif noise_type == "White Noise":
        # White noise looks like uniform noise -> use averaging filter
        denoised = cv2.blur(img_gray, (5, 5))

    elif noise_type == "Multiplicative Noise":
        # Log transform + Gaussian filter
        img_float = img_gray.astype(np.float32) / 255.0
        log_img = np.log1p(img_float)
        filtered = gaussian_filter(log_img, sigma=1)
        denoised = np.expm1(filtered)
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

    elif noise_type == "Quantization Noise":
        # Bilateral filter preserves edges while smoothing
        denoised = cv2.bilateralFilter(img_gray, 9, 75, 75)

    else:
        denoised = img_gray

    return denoised

if uploaded_file is not None:
    # Read image
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Remove selected noise
    denoised_img = remove_noise(image_bgr, noise_type)

    # Show images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(denoised_img, caption=f"Denoised ({noise_type})", use_container_width=True, channels="GRAY")
