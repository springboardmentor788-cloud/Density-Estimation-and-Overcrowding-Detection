import streamlit as st
import cv2
import numpy as np
import torch
import time
from collections import deque
from models.csrnet import CSRNet

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="CrowdVision", layout="wide")

# ------------------ STYLE ------------------
st.markdown("""
<style>
body { background-color: #0f172a; }
.card {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 12px;
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚨 CrowdVision Dashboard</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Control Panel")

threshold_warning = st.sidebar.slider("Warning Level", 50, 300, 100)
threshold_danger = st.sidebar.slider("Danger Level", 100, 500, 200)

uploaded_file = st.sidebar.file_uploader(
    "Upload Image / Video",
    type=["jpg", "jpeg", "png", "mp4"],
    accept_multiple_files=False
)

start = st.sidebar.button("▶ Start Monitoring")

if uploaded_file is None:
    st.sidebar.info("Upload an image or video to begin.")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load("CSRNet_Modified.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ------------------ LAYOUT ------------------
left, right = st.columns([3, 1])
video_placeholder = left.empty()
metrics_placeholder = right.empty()

# ------------------ PROCESS FUNCTION ------------------
def process_frame(frame):
    img = cv2.resize(frame, (320, 240)).astype(np.float32)

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    density_map = output.squeeze().numpy()
    density_map[density_map < 0] = 0

    count = int(density_map.sum() / 10)

    heatmap = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return count, overlay

# ------------------ MAIN ------------------
if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) == 0:
        st.error("File is empty or corrupted")
        st.stop()

    file_type = uploaded_file.type
    count_history = deque(maxlen=10)

    # ---------- IMAGE ----------
    if "image" in file_type:

        if start:
            np_arr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("Invalid image file")
                st.stop()

            count, overlay = process_frame(frame)

            # STATUS
            if count < threshold_warning:
                status = "NORMAL"
                color = "#22c55e"
            elif count < threshold_danger:
                status = "WARNING"
                color = "#facc15"
            else:
                status = "DANGER"
                color = "#ef4444"

            frame_small = cv2.resize(frame, (300, 200))
            overlay_small = cv2.resize(overlay, (300, 200))

            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            overlay_small = cv2.cvtColor(overlay_small, cv2.COLOR_BGR2RGB)

            with video_placeholder.container():
                st.markdown("### 🎥 Analysis")
                v1, v2 = st.columns(2)
                v1.image(frame_small, caption="Original")
                v2.image(overlay_small, caption="Heatmap")

            with metrics_placeholder.container():
                st.markdown("### 📊 Results")
                st.markdown(f"<div class='card'><h2>👥 {count}</h2><p>Crowd Count</p></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card'><h3 style='color:{color}'>{status}</h3><p>Status</p></div>", unsafe_allow_html=True)

    # ---------- VIDEO ----------
    else:

        if start:
            temp_path = "temp_video.mp4"

            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            cap = cv2.VideoCapture(temp_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                count, overlay = process_frame(frame)

                count_history.append(count)
                smooth_count = int(sum(count_history) / len(count_history))

                # STATUS
                if smooth_count < threshold_warning:
                    status = "NORMAL"
                    color = "#22c55e"
                elif smooth_count < threshold_danger:
                    status = "WARNING"
                    color = "#facc15"
                else:
                    status = "DANGER"
                    color = "#ef4444"

                frame_small = cv2.resize(frame, (300, 200))
                overlay_small = cv2.resize(overlay, (300, 200))

                frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                overlay_small = cv2.cvtColor(overlay_small, cv2.COLOR_BGR2RGB)

                with video_placeholder.container():
                    st.markdown("### 🎥 Live Monitoring")
                    v1, v2 = st.columns(2)
                    v1.image(frame_small, caption="Original")
                    v2.image(overlay_small, caption="Heatmap")

                with metrics_placeholder.container():
                    st.markdown("### 📊 Live Metrics")

                    st.markdown(f"<div class='card'><h2>👥 {smooth_count}</h2><p>Crowd Count</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card'><h3 style='color:{color}'>{status}</h3><p>Status</p></div>", unsafe_allow_html=True)

                    if status == "DANGER":
                        st.error("🚨 OVERCROWDING DETECTED!")
                    elif status == "WARNING":
                        st.warning("⚠️ Crowd Increasing")
                    else:
                        st.success("✅ All Normal")

                time.sleep(0.03)

            cap.release()