# 👁️ DeepVision

**DeepVision** is a comprehensive, AI-powered crowd analytics and density estimation system. Built using **CSRNet**, DeepVision enables real-time monitoring of crowd density, providing insightful analytics and automated alert notifications for overcrowding scenarios. 

This project encompasses a full pipeline: from model training and evaluation to a responsive, dynamic web-based dashboard for real-time monitoring.

---

## 🌟 Key Features

* **Real-time Live Monitoring**: Process webcam feeds or uploaded videos (MP4/AVI) to estimate crowd density in real-time.
* **Modern Analytics Dashboard**: A pristine, glassmorphic Streamlit interface featuring live metrics, historical data analytics, and real-time visualization overlays.
* **Overcrowding Alerts**: Automated SMS and email notifications (via Twilio/SMTP) triggered when crowd thresholds are exceeded.
* **Hardware Acceleration**: Built-in compatibility with NVIDIA CUDA for high-performance GPU tensor processing.
* **End-to-End Pipeline**: Scripts for dataset preprocessing, model training (`train.py`), pipeline evaluation (`evaluate.py`), and visualization (`visualize.py`).
* **Docker Ready**: Fully containerized using Docker and Docker Compose for easy, consistent, and scalable deployment across diverse environments.

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit Dashboard application
├── alert.py                   # Twilio/SMTP alert notification system
├── csrnet.py                  # PyTorch implementation of the CSRNet architecture
├── train.py                   # Training script for the CSRNet model
├── evaluate.py                # Model evaluation and performance metrics
├── live_video_inference.py    # Standalone CLI real-time webcam inference
├── video_inference.py         # Standalone CLI mp4 video file inference
├── data_loader.py             # PyTorch Custom Dataset loaders
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Docker specifications for containerization
├── docker-compose.yml         # Multi-container orchestration
└── .env.example               # Template for environment variables (API keys)
```

## 🛠️ Installation & Setup

### Option 1: Local Virtual Environment

1. **Clone and Setup**
   ```bash
   git clone <repository_url>
   cd "deepvision project 2"
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Rename `.env.example` to `.env` and fill in your Twilio/SMTP details for the alert system to function.
   ```text
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_FROM_NUMBER=+1234567890
   TWILIO_TO_NUMBER=+0987654321
   ```

4. **Ensure pre-trained weights reside in the root directory**
   Ensure `csrnet.pth` is located in the primary project folder.

### Option 2: Docker Deployment

To launch the dashboard using Docker Compose:
```bash
docker-compose up --build -d
```
The application will be accessible at `http://localhost:8501`.

## 🚀 Usage

### Launching the Dashboard

To start the interactive Streamlit Dashboard:
```bash
streamlit run app.py
```
**Dashboard Features:**
* **Dashboard Tab:** View the live density heatmap overlay, real-time metrics, and active alert statuses.
* **History Tab:** View interactive line charts representing session crowd history drops and averages to monitor patterns.
* **Settings Tab:** Adjust alert capacity thresholds and pick your desired video source natively.

### Running Individual CLI Tools
For terminal deployment without a UI frontend, use:
* **Webcam Feed:** `python live_video_inference.py`
* **Video File:** `python video_inference.py --video your_video.mp4`
* **Evaluate Accuracy:** `python evaluate.py`

## 🧠 Model Architecture
DeepVision employs **CSRNet (Congested Scene Recognition Network)**. It utilizes the first 10 layers of VGG-16 as a front-end for feature extraction, combined with dilated convolutional layers in the back-end to enlarge the receptive fields and extract deeper contextual information without losing resolution, generating highly accurate density maps.

## 🤝 Contribution
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.
