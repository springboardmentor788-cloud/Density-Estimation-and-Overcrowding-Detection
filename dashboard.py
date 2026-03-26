import streamlit as st
import cv2
import torch
import numpy as np
from models.csrnet import CSRNet

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="DeepVision Crowd Monitor", layout="wide")

st.title(" DeepVision Crowd Monitor")
st.markdown("### Real-Time Crowd Density & Overcrowding Detection")

# --------------------------------
# Load Model
# --------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

ALERT_THRESHOLD = 200

# --------------------------------
# Camera
# --------------------------------
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
count_display = st.empty()
alert_display = st.empty()

cap = cv2.VideoCapture(0)

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame_resized = cv2.resize(frame, (640, 480))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_np = img_rgb.astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        density_map = output.cpu().numpy()[0, 0]
        count = density_map.sum()

        # Normalize density map
        density_map = cv2.resize(density_map, (640, 480))
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        density_map = density_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

        FRAME_WINDOW.image(heatmap, channels="BGR")

        count_display.metric("👥 Crowd Count", int(count))

        if count > ALERT_THRESHOLD:
            alert_display.error(" ALERT: Overcrowded!")
        else:
            alert_display.success(" Safe Crowd Level")

else:
    cap.release()