import streamlit as st
import cv2
import numpy as np
import time
from predict import predict_count, load_model

st.set_page_config(
    page_title="DeepVision Crowd Monitor",
    page_icon="👁",
    layout="wide"
)

# ---------- UI STYLE ----------
st.markdown("""
<style>

/* Base */
.stApp {
    background: #020617;
    color: white;
}

/* Header */
.header-card {
    background: linear-gradient(90deg, #0f172a, #1e1b4b);
    padding: 20px 24px;
    border-radius: 14px;
    border: 0.5px solid #334155;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.logo-circle {
    width: 46px;
    height: 46px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.header-text h2 {
    margin: 0;
    color: #f1f5f9;
    font-size: 20px;
    font-weight: 600;
}
.header-text p {
    margin: 2px 0 0 0;
    color: #94a3b8;
    font-size: 13px;
}
.live-badge {
    margin-left: auto;
    background: #052e16;
    border: 0.5px solid #15803d;
    color: #86efac;
    font-size: 12px;
    padding: 5px 14px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}
.live-dot {
    width: 7px;
    height: 7px;
    background: #4ade80;
    border-radius: 50%;
    animation: blink 1.8s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Alert banners */
.alert-error {
    background: #1c0a0a;
    border: 0.5px solid #991b1b;
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 11px 16px;
    color: #fca5a5;
    font-size: 14px;
    margin: 4px 0;
}
.alert-ok {
    background: #052e16;
    border: 0.5px solid #15803d;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 11px 16px;
    color: #86efac;
    font-size: 14px;
    margin: 4px 0;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #0f172a;
    border: 0.5px solid #1e293b;
    padding: 14px;
    border-radius: 10px;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 28px !important;
}

/* Panel labels */
.panel-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
}
.badge-blue  { background: #1e3a5f; color: #93c5fd; }
.badge-purple{ background: #2e1065; color: #c4b5fd; }
.badge-red   { background: #3b0a0a; color: #fca5a5; }
.badge-green { background: #052e16; color: #86efac; }

/* Buttons */
.stButton > button {
    background-color: #0f172a;
    color: #94a3b8;
    border-radius: 8px;
    border: 0.5px solid #1e293b;
    font-size: 13px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #1e293b;
    color: #f1f5f9;
    border-color: #334155;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f172a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 0.5px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #64748b;
    border-radius: 7px;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #c7d2fe !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #070f1f;
    border-right: 0.5px solid #1e293b;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Dividers */
hr {
    border-color: #1e293b;
}

/* Chart */
.stLineChart {
    background: #0f172a !important;
    border-radius: 10px;
    border: 0.5px solid #1e293b;
    padding: 8px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f172a;
    border: 0.5px dashed #334155;
    border-radius: 10px;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding-top: 6px;
}

</style>
""", unsafe_allow_html=True)


# ---------- SESSION STATE ----------
if "run" not in st.session_state:
    st.session_state.run = False

if "source" not in st.session_state:
    st.session_state.source = None


# ---------- SIDEBAR ----------
st.sidebar.markdown("### ⚙️ Runtime Settings")

threshold = st.sidebar.slider(
    "Overcrowding Threshold",
    0, 200, 50,
    help="Alert fires when crowd count exceeds this value"
)

alert = st.sidebar.toggle("Enable Alert", True)

st.sidebar.divider()

st.sidebar.markdown("### 📷 Camera")

camera = st.sidebar.number_input(
    "Camera Index",
    0, 5, 0,
    help="0 = default webcam"
)

st.sidebar.divider()

st.sidebar.markdown(
    "<div style='color:#475569;font-size:12px;'>DeepVision v1.0 &nbsp;·&nbsp; Real-time CSRNet</div>",
    unsafe_allow_html=True
)


# ---------- HEADER ----------
st.markdown("""
<div class="header-card">
    <div class="logo-circle">👁</div>
    <div class="header-text">
        <h2>DeepVision Crowd Monitor</h2>
        <p>Real-time Crowd Density Estimation System</p>
    </div>
    <div class="live-badge">
        <div class="live-dot"></div> Live
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📹  Live Footage", "🖼  Upload Image", "🎬  Upload Video"])


# ---------- CONTROLS ----------
c1, c2, c3, c4, c5 = st.columns([1, 2, 1, 1, 1])

with tab1:
    if c1.button("▶  Webcam", use_container_width=True):
        st.session_state.run = True
        st.session_state.source = "webcam"

with tab2:
    image_file = c2.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    run_image = c3.button("▶  Run Image", use_container_width=True)

with tab3:
    upload = c2.file_uploader(
        "Upload Video",
        label_visibility="collapsed"
    )
    if c3.button("▶  Run Video", use_container_width=True) and upload:
        st.session_state.run = True
        st.session_state.source = "video"

if c4.button("⏹  Stop", use_container_width=True):
    st.session_state.run = False

snapshot = c5.button("📷  Snapshot", use_container_width=True)

st.divider()


# ---------- STATS ----------
s1, s2, s3, s4 = st.columns(4)

frames_box = s1.empty()
avg_box    = s2.empty()
max_box    = s3.empty()
fps_box    = s4.empty()

frames_box.metric("🎞 Frames",  0)
avg_box.metric("📊 Average",    0)
max_box.metric("⬆ Peak Count", 0)
fps_box.metric("⚡ FPS",        0)


# ---------- ALERT ----------
alert_banner = st.empty()
alert_banner.markdown(
    '<div class="alert-ok">✔ &nbsp;<strong>System Ready</strong> — Waiting for input</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------- PANEL TITLES ----------
t1, t2, t3 = st.columns(3)

t1.markdown(
    '<div class="panel-label">INPUT &nbsp;<span class="badge badge-blue">Camera 0</span></div>',
    unsafe_allow_html=True
)
t2.markdown(
    '<div class="panel-label">OVERLAY &nbsp;<span class="badge badge-purple">Density</span></div>',
    unsafe_allow_html=True
)
t3.markdown(
    '<div class="panel-label">DENSITY MAP &nbsp;<span class="badge badge-red">Heatmap</span></div>',
    unsafe_allow_html=True
)


# ---------- PANELS ----------
col1, col2, col3 = st.columns(3)

input_panel   = col1.empty()
overlay_panel = col2.empty()
density_panel = col3.empty()

st.divider()


# ---------- CHART ----------
st.markdown(
    '<div style="color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">📈 Crowd Count — Live Feed</div>',
    unsafe_allow_html=True
)
chart = st.line_chart()


# ---------- LOAD MODEL ----------
model, device = load_model()

data        = []
max_count   = 0
frame_count = 0


# ---------- PROCESS FRAME ----------
def process(frame):
    count, density = predict_count(frame, model, device)

    if hasattr(density, "cpu"):
        density = density.squeeze().cpu().numpy()

    density = cv2.resize(density, (frame.shape[1], frame.shape[0]))

    norm = density.copy()
    if norm.max() > 0:
        norm = norm / norm.max()

    heatmap = cv2.applyColorMap(
        (norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return count, overlay, heatmap


def show_alert(count, threshold, alert_enabled):
    if alert_enabled and count > threshold:
        alert_banner.markdown(
            f'<div class="alert-error">⚠ &nbsp;<strong>Overcrowding Detected</strong>'
            f' — Current count <strong>({int(count)})</strong> exceeds threshold ({threshold})</div>',
            unsafe_allow_html=True
        )
    else:
        alert_banner.markdown(
            f'<div class="alert-ok">✔ &nbsp;<strong>Normal Crowd</strong>'
            f' — Count ({int(count)}) is within safe limits</div>',
            unsafe_allow_html=True
        )


# ---------- IMAGE RUN ----------
if image_file and run_image:

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    count, overlay, density = process(frame)

    input_panel.image(frame,   channels="BGR", use_container_width=True)
    overlay_panel.image(overlay, channels="BGR", use_container_width=True)
    density_panel.image(density, channels="BGR", use_container_width=True)

    frames_box.metric("🎞 Frames",  1)
    avg_box.metric("📊 Average",    int(count))
    max_box.metric("⬆ Peak Count", int(count))

    show_alert(count, threshold, alert)


# ---------- LIVE / VIDEO RUN ----------
if st.session_state.run:

    if st.session_state.source == "webcam":
        cap = cv2.VideoCapture(camera)

    elif st.session_state.source == "video":
        tfile = open("temp.mp4", "wb")
        tfile.write(upload.read())
        tfile.close()
        cap = cv2.VideoCapture("temp.mp4")

    prev = time.time()

    while st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        count, overlay, density_map = process(frame)

        data.append(count)
        avg       = sum(data) / len(data)
        max_count = max(data)

        frames_box.metric("🎞 Frames",  frame_count)
        avg_box.metric("📊 Average",    int(avg))
        max_box.metric("⬆ Peak Count", int(max_count))

        now  = time.time()
        fps  = 1 / (now - prev)
        prev = now
        fps_box.metric("⚡ FPS", int(fps))

        show_alert(count, threshold, alert)

        input_panel.image(frame,       channels="BGR", use_container_width=True)
        overlay_panel.image(overlay,   channels="BGR", use_container_width=True)
        density_panel.image(density_map, channels="BGR", use_container_width=True)

        chart.add_rows([count])

        if snapshot:
            cv2.imwrite("snapshot.jpg", overlay)
            st.toast("📷 Snapshot saved!", icon="✅")

        time.sleep(0.03)

    cap.release()