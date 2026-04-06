import streamlit as st
import cv2
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from models.csrnet import CSRNet

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="DeepVision AI", layout="wide")

# =========================================
# GLOBAL STYLE (FULL UI THEME)
# =========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

/* Title */
.main-title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    background: linear-gradient(90deg,#22c55e,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#22c55e,#06b6d4);
    color:white;
    border-radius:10px;
    height:3em;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown('<div class="main-title">🚀 DeepVision Crowd Intelligence</div>', unsafe_allow_html=True)
st.markdown("### AI-powered Crowd Monitoring & Analytics System")

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("models/csrnet_main.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# =========================================
# SIDEBAR NAVIGATION
# =========================================
page = st.sidebar.radio("Navigation", ["🏠 Dashboard", "📊 Analytics", "ℹ️ About"])

# =========================================
# STORAGE
# =========================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================
# DASHBOARD PAGE
# =========================================
if page == "🏠 Dashboard":

    st.sidebar.header("Controls")
    threshold = st.sidebar.slider("Alert Threshold", 50, 500, 200)
    heatmap_toggle = st.sidebar.checkbox("Show Heatmap", True)

    mode = st.radio("Select Mode", ["Upload Image", "Live Camera"])

    # -------------------------
    # IMAGE MODE
    # -------------------------
    if mode == "Upload Image":

        file = st.file_uploader("Upload Image")

        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_np = img_rgb.astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)

            density = output.cpu().numpy()[0, 0]
            count = int(density.sum())

            st.image(img_rgb, caption="Input Image")

            if heatmap_toggle:
                heatmap = cv2.resize(density, (img.shape[1], img.shape[0]))
                heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap = heatmap.astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                st.image(heatmap, caption="Density Map")

            st.metric("👥 Crowd Count", count)

            # Save history
            st.session_state.history.append({
                "time": datetime.now(),
                "count": count
            })

            if count > threshold:
                st.error("🚨 Overcrowded!")
            else:
                st.success("Safe")

    # -------------------------
    # LIVE CAMERA
    # -------------------------
    else:
        start = st.button("Start Camera")

        FRAME = st.image([])

        if start:
            cap = cv2.VideoCapture(0)
            prev = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = torch.tensor(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img)

                density = output.cpu().numpy()[0, 0]
                count = int(density.sum())

                if heatmap_toggle:
                    heatmap = cv2.resize(density, (640, 480))
                    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap = heatmap.astype(np.uint8)
                    frame = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # FPS
                now = time.time()
                fps = 1/(now-prev) if prev!=0 else 0
                prev = now

                cv2.putText(frame, f"Count: {count}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

                FRAME.image(frame, channels="BGR")

                st.session_state.history.append({
                    "time": datetime.now(),
                    "count": count
                })

# =========================================
# ANALYTICS PAGE
# =========================================
elif page == "📊 Analytics":

    st.subheader("📈 Crowd Analytics")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        st.line_chart(df["count"])

        st.write("Recent Data")
        st.dataframe(df.tail(10))

    else:
        st.info("No data available yet")

# =========================================
# ABOUT PAGE
# =========================================
else:
    st.subheader("ℹ️ About Project")

    st.markdown("""
    **DeepVision Crowd Intelligence System**

    🔹 Built using CSRNet Deep Learning Model  
    🔹 Real-time crowd monitoring  
    🔹 Density estimation & heatmaps  
    🔹 Alert system for overcrowding  

    **Tech Stack:**
    - PyTorch
    - OpenCV
    - Streamlit

    🚀 Designed for Smart Cities & Surveillance Systems
    """)