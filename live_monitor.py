import streamlit as st
import cv2
import torch
import numpy as np
from models.csrnet import CSRNet   # Ensure path is correct

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="DeepVision Crowd Monitor",
    layout="wide"
)

st.title(" DeepVision Crowd Monitor")
st.subheader("Real-Time Crowd Density & Overcrowding Detection")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load("csrnet_trained.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# ALERT THRESHOLD SLIDER
# --------------------------------------------------
threshold = st.slider(
    " Set Overcrowding Threshold",
    min_value=1,
    max_value=20,
    value=5
)

# --------------------------------------------------
# START CAMERA
# --------------------------------------------------
start = st.checkbox("Start Camera")

if start:

    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()
    metric_placeholder = st.empty()

    while True:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected")
            break

        # -------------------------
        # PREPARE IMAGE
        # -------------------------
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0)

        # -------------------------
        # MODEL PREDICTION
        # -------------------------
        with torch.no_grad():
            output = model(img)

        density_map = output.squeeze().cpu().numpy()

        # Remove negative noise
        density_map[density_map < 0] = 0

        # Count people
        count = float(density_map.sum())

        # Normalize for heatmap
        if density_map.max() > 0:
            density_map = density_map / density_map.max()

        density_map = cv2.resize(
            density_map,
            (frame.shape[1], frame.shape[0])
        )

        heatmap = cv2.applyColorMap(
            np.uint8(255 * density_map),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            frame,
            0.6,
            heatmap,
            0.4,
            0
        )

        # -------------------------
        # UPDATE DASHBOARD
        # -------------------------
        frame_placeholder.image(
            overlay,
            channels="BGR",
            use_container_width=True
        )

        with metric_placeholder.container():

            st.metric("👥 Crowd Count", int(count))

            if count > threshold:
                st.error(" OVERCROWDED AREA DETECTED!")
            elif count > threshold * 0.6:
                st.warning("⚠ Moderate Crowd Level")
            else:
                st.success(" Safe Crowd Level")

    cap.release()