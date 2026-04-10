🤖 AI Crowd Monitoring Dashboard

Real-time Crowd Density Estimation using Deep Learning (CSRNet)

📌 Overview

This project is an AI-based Crowd Monitoring System that estimates crowd density from:

🎥 Live Webcam
📁 Video Upload
🖼 Image Upload

It generates:

Crowd Count
Density Map
Heatmap Overlay
Overcrowding Alert
Live Dashboard Visualization

The system is designed for smart surveillance, drone monitoring, and public safety.

🚀 Features
🎥 Real-time Monitoring
Live webcam crowd counting
Real-time density estimation
FPS tracking
📁 Video Processing
Upload video file
Frame-by-frame crowd detection
Heatmap overlay
🖼 Image Prediction
Upload crowd image
Density map generation
Instant crowd count
📊 Interactive Dashboard
Modern Streamlit UI
3-panel layout (Input | Overlay | Density)
Live graph visualization
Session statistics
🚨 Overcrowding Detection
Adjustable threshold
Alert banner
Real-time status monitoring
📈 Analytics
Average crowd count
Max crowd count
Frames processed
FPS display
📸 Snapshot
Save current frame
Export overlay image
🧠 Model Used

CSRNet — Convolutional Neural Network for Crowd Counting

Architecture:

VGG16 Frontend
Dilated Convolution Backend
Density Map Regression
🖥 Dashboard Layout
DeepVision Crowd Monitor Dashboard

Live | Upload Image | Upload Video

INPUT | OVERLAY | DENSITY MAP

Frames | Avg | Max | FPS

Overcrowding Alert
📂 Project Structure
project/
│
├── dashboard.py
├── main.py
├── predict.py
├── train.py
├── dataset.py
│
├── csrnet_model.pth
│
├── outputs/
│
└── README.md
⚙️ Installation
1. Clone Repository
git clone https://github.com/yourusername/repo-name.git
cd repo-name
2. Install Dependencies
pip install -r requirements.txt

Or manually:

pip install torch torchvision
pip install opencv-python
pip install streamlit
pip install numpy
▶️ Run Dashboard
streamlit run dashboard.py

Open browser:

http://localhost:8501
🎮 Usage
Webcam
Click Webcam
Live crowd detection starts
Upload Video
Go to Upload Video
Select file
Click Run
Upload Image
Go to Upload Image
Upload image
Click Run Image
📊 Output

The dashboard displays:

Crowd Count
Density Heatmap
Overlay Visualization
FPS
Average Count
Max Count
Alert Status
🚨 Overcrowding Detection

System triggers alert when:

Crowd Count > Threshold

Threshold adjustable from sidebar.

📈 Applications
Smart City Surveillance
Drone Crowd Monitoring
Event Management
Traffic Monitoring
Stadium Crowd Control
Mall Footfall Analysis
🧪 Future Improvements
Multi-camera support
Zone-based counting
Object detection integration
YOLO + CSRNet hybrid
Database logging
Deployment on cloud
👨‍💻 Author

Tarun Gupta
AI / ML Developer
Crowd Monitoring System Project