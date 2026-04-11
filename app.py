import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from models.csrnet import CSRNet
import config

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="DeepVision Analytics Pro", layout="wide")

# ---------------- UI STYLING ---------------- #
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .dashboard-title { 
        font-family: 'Inter', sans-serif; 
        font-size: 32px; 
        font-weight: 700; 
        color: #00f2ff; 
        letter-spacing: 1px;
    }
    .status-container {
        padding: 15px;
        border-radius: 4px;
        text-align: center;
        font-family: monospace;
        font-size: 16px;
        border: 1px solid #1e293b;
        margin-top: 10px;
    }
    .status-safe { border-left: 5px solid #10b981; background: #064e3b22; color: #34d399; }
    .status-alert { border-left: 5px solid #ef4444; background: #7f1d1d22; color: #f87171; }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    model_path = "best_model.pth" if os.path.exists("best_model.pth") else "models/csrnet_main.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ---------------- PROCESSING ---------------- #
def predict(image):
    img = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        output = model(tensor)
    
    density = output.squeeze().cpu().numpy()
    
    # FIX: Remove negative values by clipping at 0
    density = np.maximum(density, 0)
    count = int(density.sum())
    
    rescaled = cv2.resize(density, (image.shape[1], image.shape[0]))
    norm = cv2.normalize(rescaled, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(np.uint8(norm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    return count, heatmap, overlay

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("### SYSTEM CONTROLS")
    current_mode = st.radio("ANALYSIS MODE", ["Image Analysis", "Video Intelligence", "Live Stream"], index=0)
    st.divider()
    
    st.markdown("### PERFORMANCE")
    frame_skip = st.select_slider("OPTIMIZATION (FRAME SKIP)", options=[1, 2, 3, 5, 10], value=3)
    st.divider()
    threshold = st.slider("CROWD THRESHOLD", 10, 1000, 200)

# ---------------- MAIN DASHBOARD ---------------- #
st.markdown('<p class="dashboard-title">DEEPVISION ANALYTICS ENGINE</p>', unsafe_allow_html=True)

source_file = None
run_cam = False

if current_mode == "Image Analysis":
    source_file = st.file_uploader("UPLOAD IMAGE", type=["jpg", "jpeg", "png"])
elif current_mode == "Video Intelligence":
    source_file = st.file_uploader("UPLOAD VIDEO", type=["mp4"])
else:
    # Use session state to monitor the checkbox toggle
    run_cam = st.checkbox("ACTIVATE SENSOR", key="webcam_toggle")

m_col1, m_col2 = st.columns([1, 3])
count_metric = m_col1.empty()
status_metric = m_col2.empty()

st.divider()

f_col1, f_col2, f_col3 = st.columns(3)
p1, p2, p3 = f_col1.empty(), f_col2.empty(), f_col3.empty()

# ---------------- EXECUTION LOGIC ---------------- #
if source_file or (current_mode == "Live Stream" and run_cam):
    if current_mode == "Live Stream":
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_file.read())
        cap = cv2.VideoCapture(tfile.name)

    frame_count = 0
    last_count, last_heatmap, last_overlay = 0, None, None

    while cap.isOpened():
        # RE-CHECK: If user unchecked the box during the loop, release and rerun
        if current_mode == "Live Stream" and not st.session_state.webcam_toggle:
            cap.release()
            st.rerun() 
            
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if frame_count % frame_skip == 0:
            last_count, last_heatmap, last_overlay = predict(frame_rgb)
        
        count_metric.metric("CURRENT COUNT", last_count)
        if last_count > threshold:
            status_metric.markdown(f'<div class="status-container status-alert">ALERT: SYSTEM OVER CAPACITY</div>', unsafe_allow_html=True)
        else:
            status_metric.markdown(f'<div class="status-container status-safe">STATUS: OPERATIONAL NORMAL</div>', unsafe_allow_html=True)
        
        p1.image(frame_rgb, caption="SOURCE INPUT", use_container_width=True)
        if last_heatmap is not None:
            p2.image(last_heatmap, caption="DENSITY GRADIENT", use_container_width=True)
            p3.image(last_overlay, caption="ANALYTIC OVERLAY", use_container_width=True)
        
        frame_count += 1
        if current_mode == "Image Analysis": 
            break

    cap.release()

# Mode switch cleanup
if current_mode != st.session_state.get('last_mode'):
    st.session_state['last_mode'] = current_mode
    st.rerun()