import streamlit as st
import cv2
import torch
import numpy as np
import time
from PIL import Image
import os
from datetime import datetime
from csrnet import CSRNet
from alert import send_alert_async
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()

# Setup Layout & Aesthetics
st.set_page_config(page_title="DeepVision | Live Crowd Monitoring", layout="wide", page_icon="👁️")

# Initialization State
if "alert_cooldown" not in st.session_state:
    st.session_state.alert_cooldown = 0
if "history_data" not in st.session_state:
    st.session_state.history_data = [] # List of dicts for analytics
if "source" not in st.session_state:
    st.session_state.source = "Webcam"
if "threshold" not in st.session_state:
    st.session_state.threshold = 150

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"

@st.cache_resource
def load_model():
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("csrnet.pth", map_location=device))
    model.eval()
    return model

with st.spinner(f"Loading Model on {device_name}..."):
    model = load_model()

# Custom CSS for the new aesthetic
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up headers */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    /* Sidebar specific styling */
    [data-testid="stSidebar"] {
        background-color: #12151f !important;
        border-right: 1px solid #1f2430;
    }
    
    /* Metric Card Styling to match screenshot */
    .metric-card {
        background-color: #1f2334;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        border: 1px solid #2d3342;
    }
    .metric-icon {
        background-color: #272d42;
        color: #6a82fb;
        width: 60px;
        height: 60px;
        border-radius: 14px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 28px;
        margin-right: 20px;
        box-shadow: inset 0 0 10px rgba(106, 130, 251, 0.1);
    }
    .metric-content {
        display: flex;
        flex-direction: column;
    }
    .metric-content h4 {
        margin: 0;
        font-size: 13px;
        color: #8b949e;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding-bottom: 5px;
    }
    .metric-content h2 {
        margin: 0;
        font-size: 28px;
        color: #ffffff;
        font-weight: 700;
        line-height: 1.2;
    }
    
    /* Custom Sub-header divider */
    .header-divider hr {
        border-color: #2d3342;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .header-divider h3 {
        color: #8b949e;
        font-size: 16px;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Define Sidebar Custom Navigation
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
    <div style="font-size: 24px; color: #6a82fb;">📷</div>
    <h2 style='margin: 0; color: white;'>DeepVision</h2>
</div>
<hr style="border-color: #2d3342; margin-top: 0;">
""", unsafe_allow_html=True)

with st.sidebar:
    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "History", "Settings"],
        icons=["house", "clock-history", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#12151f"},
            "icon": {"color": "#6a82fb", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#2d3342",
            },
            "nav-link-selected": {"background-color": "#272d42"},
        },
    )


def process_frame(frame, width, height, enable_alert=True):
    img = cv2.resize(frame, (256, 256))
    img = img / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    inference_time = time.time() - start_time
    fps = 1 / inference_time if inference_time > 0 else 0

    density = output.squeeze().cpu().numpy()
    count = density.sum()

    heatmap = cv2.GaussianBlur(density, (25, 25), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = cv2.resize(heatmap, (width, height))
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    
    if count > st.session_state.threshold:
        status = "🚨 overcrowded"
        if enable_alert and time.time() - st.session_state.alert_cooldown > 15:
            send_alert_async(count)
            st.session_state.alert_cooldown = time.time()
    else:
        status = "✅ Normal"

    return overlay, count, fps, status

# Helper for card rendering
def render_metric_card(icon, title, value, value_color="#ffffff", icon_bg="#272d42"):
    return f"""
    <div class="metric-card">
        <div class="metric-icon" style="background-color: {icon_bg};">{icon}</div>
        <div class="metric-content">
            <h4>{title}</h4>
            <h2 style="color: {value_color};">{value}</h2>
        </div>
    </div>
    """


# ==========================================
# Application Routing
# ==========================================

if selected_page == "Dashboard":
    st.markdown("<h1>Live Crowd Monitoring</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header-divider">
        <h3>Live Feed & Density Map</h3>
        <hr>
    </div>
    """, unsafe_allow_html=True)
    
    video_placeholder = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Static placeholders for metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        count_placeholder = st.empty()
        count_placeholder.markdown(render_metric_card("👥", "Current Crowd Count", "--"), unsafe_allow_html=True)
    with col2:
        status_placeholder = st.empty()
        status_placeholder.markdown(render_metric_card("🔔", "Alert Status", "Connecting..."), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("🖩", "Hardware Acceleration", device_name), unsafe_allow_html=True)

    if st.session_state.source == "Webcam":
        if "process_webcam" not in st.session_state:
            st.session_state.process_webcam = False
            
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Start Webcam"):
                st.session_state.process_webcam = True
        with btn_col2:
            if st.button("Stop Webcam"):
                st.session_state.process_webcam = False
                
        if st.session_state.process_webcam:
            cap = cv2.VideoCapture(0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: 
                        st.session_state.process_webcam = False
                        break
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    overlay, count, fps, status = process_frame(frame, width, height)
                    
                    # Update Session History
                    st.session_state.history_data.append({
                        "Timestamp": datetime.now(), 
                        "Count": count, 
                        "Status": status,
                        "FPS": fps
                    })
                    if len(st.session_state.history_data) > 1000:
                        st.session_state.history_data.pop(0)
                    
                    video_placeholder.image(overlay, use_container_width=True)
                    
                    # Dynamic Metric Update
                    count_placeholder.markdown(render_metric_card("👥", "Current Crowd Count", str(int(count))), unsafe_allow_html=True)
                    
                    status_color = "#ff4b4b" if "overcrowded" in status.lower() else "#00C9FF"
                    icon_bg = "rgba(255, 75, 75, 0.15)" if "overcrowded" in status.lower() else "rgba(0, 201, 255, 0.15)"
                    status_placeholder.markdown(render_metric_card("🔔", "Alert Status", status, value_color=status_color, icon_bg=icon_bg), unsafe_allow_html=True)
                    
                    # Store last state to persist results when stopped
                    st.session_state.last_frame = overlay
                    st.session_state.last_count = count
                    st.session_state.last_status = status
            finally:
                cap.release()
        else:
            # When stopped but there is data, display the final results
            if "last_frame" in st.session_state and st.session_state.last_frame is not None:
                video_placeholder.image(st.session_state.last_frame, use_container_width=True)
                count_placeholder.markdown(render_metric_card("👥", "Final Crowd Count", str(int(st.session_state.last_count))), unsafe_allow_html=True)
                
                status = st.session_state.last_status
                status_color = "#ff4b4b" if "overcrowded" in status.lower() else "#00C9FF"
                icon_bg = "rgba(255, 75, 75, 0.15)" if "overcrowded" in status.lower() else "rgba(0, 201, 255, 0.15)"
                status_placeholder.markdown(render_metric_card("🔔", "Final Status", status, value_color=status_color, icon_bg=icon_bg), unsafe_allow_html=True)
                
                st.info("✅ Webcam Stopped. Summary of the current session is shown below.")
                
                if len(st.session_state.history_data) > 0:
                    counts = [item["Count"] for item in st.session_state.history_data]
                    
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        st.markdown(render_metric_card("📈", "Max Crowd Count", str(int(max(counts)))), unsafe_allow_html=True)
                    with rc2:
                        st.markdown(render_metric_card("📊", "Avg Crowd Count", str(int(sum(counts)/len(counts)))), unsafe_allow_html=True)
                    with rc3:
                        st.markdown(render_metric_card("🎦", "Frames Analyzed", str(len(counts))), unsafe_allow_html=True)

    elif st.session_state.source == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = "temp_uploaded_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())
                
            cap = cv2.VideoCapture(tfile)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if "process_uploaded" not in st.session_state:
                st.session_state.process_uploaded = False
                
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Process Video"):
                    st.session_state.process_uploaded = True
            with btn_col2:
                if st.button("Stop Processing"):
                    st.session_state.process_uploaded = False
            
            if st.session_state.process_uploaded:
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: 
                            st.success("Video processing completed!")
                            st.session_state.process_uploaded = False
                            break
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        overlay, count, fps, status = process_frame(frame, width, height, enable_alert=False)
                        
                        st.session_state.history_data.append({
                            "Timestamp": datetime.now(), 
                            "Count": count, 
                            "Status": status,
                            "FPS": fps
                        })
                        if len(st.session_state.history_data) > 1000:
                            st.session_state.history_data.pop(0)

                        video_placeholder.image(overlay, use_container_width=True)
                        
                        count_placeholder.markdown(render_metric_card("👥", "Current Crowd Count", str(int(count))), unsafe_allow_html=True)
                        status_color = "#ff4b4b" if "overcrowded" in status.lower() else "#00C9FF"
                        icon_bg = "rgba(255, 75, 75, 0.15)" if "overcrowded" in status.lower() else "rgba(0, 201, 255, 0.15)"
                        status_placeholder.markdown(render_metric_card("🔔", "Alert Status", status, value_color=status_color, icon_bg=icon_bg), unsafe_allow_html=True)
                        
                        # Store last state to persist results when stopped
                        st.session_state.last_frame = overlay
                        st.session_state.last_count = count
                        st.session_state.last_status = status
                finally:
                    cap.release()
            else:
                # When stopped but there is data, display the final results
                if "last_frame" in st.session_state and st.session_state.last_frame is not None:
                    video_placeholder.image(st.session_state.last_frame, use_container_width=True)
                    count_placeholder.markdown(render_metric_card("👥", "Final Crowd Count", str(int(st.session_state.last_count))), unsafe_allow_html=True)
                    
                    status = st.session_state.last_status
                    status_color = "#ff4b4b" if "overcrowded" in status.lower() else "#00C9FF"
                    icon_bg = "rgba(255, 75, 75, 0.15)" if "overcrowded" in status.lower() else "rgba(0, 201, 255, 0.15)"
                    status_placeholder.markdown(render_metric_card("🔔", "Final Status", status, value_color=status_color, icon_bg=icon_bg), unsafe_allow_html=True)
                    
                    st.info("✅ Processing Stopped. Summary of the current session is shown below.")
                    
                    if len(st.session_state.history_data) > 0:
                        counts = [item["Count"] for item in st.session_state.history_data]
                        
                        rc1, rc2, rc3 = st.columns(3)
                        with rc1:
                            st.markdown(render_metric_card("📈", "Max Crowd Count", str(int(max(counts)))), unsafe_allow_html=True)
                        with rc2:
                            st.markdown(render_metric_card("📊", "Avg Crowd Count", str(int(sum(counts)/len(counts)))), unsafe_allow_html=True)
                        with rc3:
                            st.markdown(render_metric_card("🎦", "Frames Analyzed", str(len(counts))), unsafe_allow_html=True)

    elif st.session_state.source == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png', 'webp'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            height, width = frame.shape[:2]
            
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Check if this is a new image
            if st.session_state.get("last_uploaded_image") != file_identifier:
                st.session_state.processed_image = None
                st.session_state.image_count = None
                st.session_state.image_status = None
                st.session_state.last_uploaded_image = file_identifier
                
            if st.session_state.processed_image is not None:
                video_placeholder.image(st.session_state.processed_image, use_container_width=True)
                count_placeholder.markdown(render_metric_card("👥", "Crowd Count", str(int(st.session_state.image_count))), unsafe_allow_html=True)
                
                status = st.session_state.image_status
                status_color = "#ff4b4b" if "overcrowded" in status.lower() else "#00C9FF"
                icon_bg = "rgba(255, 75, 75, 0.15)" if "overcrowded" in status.lower() else "rgba(0, 201, 255, 0.15)"
                status_placeholder.markdown(render_metric_card("🔔", "Alert Status", status, value_color=status_color, icon_bg=icon_bg), unsafe_allow_html=True)
            else:
                video_placeholder.image(frame, use_container_width=True)
            
            if st.button("Process Image"):
                with st.spinner("Analyzing crowd..."):
                    overlay, count, fps, status = process_frame(frame, width, height, enable_alert=False)
                    
                    st.session_state.processed_image = overlay
                    st.session_state.image_count = count
                    st.session_state.image_status = status
                    
                    video_placeholder.image(overlay, use_container_width=True)
                    
                    count_placeholder.markdown(render_metric_card("👥", "Crowd Count", str(int(count))), unsafe_allow_html=True)
                    status_color = "#ff4b4b" if "overcrowded" in status.lower() else "#00C9FF"
                    icon_bg = "rgba(255, 75, 75, 0.15)" if "overcrowded" in status.lower() else "rgba(0, 201, 255, 0.15)"
                    status_placeholder.markdown(render_metric_card("🔔", "Alert Status", status, value_color=status_color, icon_bg=icon_bg), unsafe_allow_html=True)
                    
                    st.session_state.history_data.append({
                        "Timestamp": datetime.now(), 
                        "Count": count, 
                        "Status": status,
                        "FPS": fps
                    })
                    if len(st.session_state.history_data) > 1000:
                        st.session_state.history_data.pop(0)

elif selected_page == "History":
    st.markdown("<h1>Analytics & History</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="header-divider">
        <h3>Real-Time Crowd Density Trends</h3>
        <hr>
    </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.history_data) > 0:
        counts = [item["Count"] for item in st.session_state.history_data]
        
        st.line_chart(counts)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="header-divider">
            <h3>Session Statistics</h3>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(render_metric_card("📈", "Max Crowd Count", str(int(max(counts)))), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card("📊", "Avg Crowd Count", str(int(sum(counts)/len(counts)))), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card("🎦", "Total Frames Analyzed", str(len(counts))), unsafe_allow_html=True)
        
        st.markdown("<br><h3>Recent Data Log</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.history_data[-50:])
    else:
        st.info("No history data available yet. Start monitoring on the Dashboard to populate analytics.")

elif selected_page == "Settings":
    st.markdown("<h1>Configuration Settings</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="header-divider">
        <h3>System Controls</h3>
        <hr>
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.source = st.radio("Media Source", ["Webcam", "Upload Video", "Upload Image"], index=["Webcam", "Upload Video", "Upload Image"].index(st.session_state.source))
    st.session_state.threshold = st.slider("Alert Threshold (Count)", min_value=10, max_value=1000, value=st.session_state.threshold, step=10)
    
    st.markdown("---")
    st.markdown("### Hardware Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Device:** `{device_name}`")
        if torch.cuda.is_available():
            st.markdown(f"**CUDA Device Name:** `{torch.cuda.get_device_name(0)}`")
    with c2:
        st.markdown(f"**PyTorch Version:** `{torch.__version__}`")
        if torch.cuda.is_available():
            st.markdown(f"**CUDA Version:** `{torch.version.cuda}`")

    st.markdown("---")
    if st.button("Clear History Data"):
        st.session_state.history_data = []
        st.success("History data cleared.")
