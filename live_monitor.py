import streamlit as st
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from models.csrnet import CSRNet

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="DeepVision AI Monitor", layout="wide")

# =============================
# UI STYLE (PROFESSIONAL)
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg,#22c55e,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚀 DeepVision Crowd Monitor</div>', unsafe_allow_html=True)
st.markdown("### 🎯 YOLO  + CSRNet (Density Map)")

# =============================
# SIDEBAR
# =============================
st.sidebar.header(" Controls")

threshold = st.sidebar.slider(" Overcrowding Threshold", 1, 500, 50)

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CSRNet
    csrnet = CSRNet().to(device)
    csrnet.load_state_dict(torch.load("models/csrnet_main.pth", map_location=device))
    csrnet.eval()

    # YOLO
    yolo = YOLO("yolov8n.pt")

    return csrnet, yolo, device

csrnet, yolo, device = load_models()
st.success(" Models Loaded")

# =============================
# BUTTON
# =============================
start = st.button("▶ Start Camera")

# =============================
# LAYOUT (FIXED ERROR HERE)
# =============================
col1, col2 = st.columns(2)

frame_placeholder = col1.empty()
density_placeholder = col2.empty()

colA, colB, colC = st.columns(3)
count_box = colA.empty()
fps_box = colB.empty()
status_box = colC.empty()

alert_box = st.empty()

# =============================
# CAMERA LOOP
# =============================
if start:

    cap = cv2.VideoCapture(0)
    prev_time = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            st.error(" Camera not detected")
            break

        frame = cv2.resize(frame, (640, 480))

        # =============================
        # YOLO (PERSON DETECTION)
        # =============================
        results = yolo(frame, verbose=False)[0]

        person_count = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # =============================
        # CSRNET (DENSITY)
        # =============================
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = csrnet(img)

        density = output.squeeze().cpu().numpy()
        density[density < 0] = 0

        density_count = int(density.sum())

        # =============================
        # HYBRID COUNT (FIXED ISSUE)
        # =============================
        if person_count <= 10:
            count = person_count   # accurate small crowd
        else:
            count = density_count  # large crowd

        # =============================
        # DENSITY MAP (COLOR)
        # =============================
        density_norm = density.copy()

        if density_norm.max() > 0:
            density_norm = density_norm / density_norm.max()

        density_resized = cv2.resize(density_norm, (640, 480))

        density_color = cv2.applyColorMap(
            (density_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET   #  Blue-Yellow-Red
        )

        # =============================
        # FPS
        # =============================
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # =============================
        # ALERT SYSTEM
        # =============================
        if count > threshold:
            status = "🚨 Overcrowded"
            alert_box.error("🚨 HIGH DENSITY ALERT!")
        elif count > threshold * 0.6:
            status = "⚠ Moderate"
            alert_box.warning("⚠ Moderate Crowd")
        else:
            status = "✅ Safe"
            alert_box.success("✔ Safe Crowd")

        # =============================
        # DISPLAY OUTPUT
        # =============================
        frame_placeholder.image(frame, channels="BGR")
        density_placeholder.image(density_color, channels="BGR")

        count_box.metric("👥 Crowd Count", int(count))
        fps_box.metric("⚡ FPS", int(fps))
        status_box.metric(" Status", status)

    cap.release()
    cv2.destroyAllWindows()