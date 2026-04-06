import streamlit as st
import torch
import numpy as np
import cv2
import time
import os
from PIL import Image
from ultralytics import YOLO
from models.csrnet import CSRNet

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="DeepVision Pro UI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
.title {
    font-size: 38px;
    text-align: center;
    font-weight: bold;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚀 DeepVision Smart Crowd Monitor</div>', unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csrnet = CSRNet().to(device)
    csrnet.load_state_dict(torch.load("models/csrnet_main.pth", map_location=device))
    csrnet.eval()

    yolo = YOLO("yolov8m.pt")

    return csrnet, yolo, device

csrnet, yolo, device = load_models()

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # =========================
    # YOLO
    # =========================
    results = yolo(img_np, verbose=False)[0]
    yolo_count = 0

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            yolo_count += 1

    # =========================
    # CSRNET
    # =========================
    img = img_np.astype(np.float32)/255.0
    img = np.transpose(img,(2,0,1))
    img = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = csrnet(img)

    density = output.squeeze().cpu().numpy()
    density[density < 0] = 0
    density_count = int(density.sum())

    final_count = density_count if yolo_count > 20 else yolo_count

    # =========================
    # DENSITY MAP
    # =========================
    d = density.copy()
    if d.max() > 0:
        d = d / d.max()

    d = cv2.resize(d, (img_np.shape[1], img_np.shape[0]))

    heatmap = cv2.applyColorMap(
        (d * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # =========================
    # OVERLAY
    # =========================
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # =========================
    # HOTSPOTS
    # =========================
    hotspot = (d > 0.6).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hotspot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)

    # =========================
    # MAIN DISPLAY
    # =========================
    st.image(overlay, caption="🔥 Crowd Analysis View", use_container_width=True)

    # =========================
    # LOWER PANEL
    # =========================
    col1, col2 = st.columns([2,1])

    with col1:
        st.image(heatmap, caption="📊 Density Map", use_container_width=True)

    with col2:
        st.markdown("### 📈 Analytics")

        st.metric("👥 Final Count", final_count)
        st.metric("YOLO Count", yolo_count)
        st.metric("CSRNet Count", density_count)

        if final_count > 300:
            st.error("🚨 HIGH DENSITY ALERT")
        elif final_count > 150:
            st.warning("⚠ Moderate Crowd")
        else:
            st.success("✅ Safe")

else:
    st.info("Upload image to start")