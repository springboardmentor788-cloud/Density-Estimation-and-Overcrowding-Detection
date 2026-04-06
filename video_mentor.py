import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import time
from models.csrnet import CSRNet

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="CCTV AI Monitor", layout="wide")
st.title("🎥 DeepVision CCTV Crowd Analysis")

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("models/csrnet_main.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

threshold = st.sidebar.slider("🚨 Alert Threshold", 50, 1000, 300)

#  IMPORTANT SCALE FIX
DENSITY_SCALE = 100  # adjust between 50–200 if needed

video_file = st.file_uploader("Upload CCTV Video", type=["mp4", "avi"])

if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_window = st.image([])
    metric_box = st.empty()

    prev_time = 0

    st.success("Processing...")

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------
        # RESIZE (VERY IMPORTANT)
        # -------------------------
        frame = cv2.resize(frame, (640, 480))

        # -------------------------
        # PREPROCESS
        # -------------------------
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(device)

        # -------------------------
        # MODEL
        # -------------------------
        with torch.no_grad():
            output = model(img)

        density = output.squeeze().cpu().numpy()
        density[density < 0] = 0

        # =========================
        # ✅ FIXED COUNT
        # =========================
        raw_count = density.sum()
        count = int(raw_count * DENSITY_SCALE)

        # -------------------------
        # NORMALIZE DENSITY
        # -------------------------
        density_norm = density.copy()
        if density_norm.max() > 0:
            density_norm = density_norm / density_norm.max()

        density_resized = cv2.resize(density_norm, (640, 480))

        # -------------------------
        # PEAK POINTS (BETTER)
        # -------------------------
        points = np.where(density_resized > 0.4)

        dot_img = frame.copy()

        for y, x in zip(points[0], points[1]):
            val = density_resized[y, x]

            if val > 0.7:
                color = (0, 0, 255)  # red (high)
            else:
                color = (0, 255, 0)  # green

            cv2.circle(dot_img, (x, y), 3, color, -1)

        # -------------------------
        # DENSITY MAP
        # -------------------------
        density_display = (density_resized * 255).astype(np.uint8)
        density_display = cv2.applyColorMap(density_display, cv2.COLORMAP_JET)

        # -------------------------
        # COMBINE
        # -------------------------
        combined = np.hstack((dot_img, density_display))

        # -------------------------
        # FPS
        # -------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # -------------------------
        # TEXT
        # -------------------------
        cv2.putText(combined, f"Crowd Count: {count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

        cv2.putText(combined, f"FPS: {int(fps)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # ALERT
        if count > threshold:
            cv2.rectangle(combined, (900, 20), (1270, 100), (0, 0, 255), -1)
            cv2.putText(combined, "HIGH DENSITY ALERT",
                        (910, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

        # -------------------------
        # DISPLAY
        # -------------------------
        frame_window.image(combined, channels="BGR")
        metric_box.metric("👥 Crowd Count", count)

    cap.release()

    st.success("✅ Completed")